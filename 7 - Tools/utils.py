from typing import TypedDict, Annotated, Literal
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.checkpoint.sqlite import SqliteSaver
# ======================== RAG ===================================
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool

from dotenv import load_dotenv
import logging
import sqlite3
import os

# Load environment variables
load_dotenv()

# ========================== Setup logging ===========================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log LangSmith configuration
logger.info(f"LangSmith Tracing: {os.environ.get('LANGCHAIN_TRACING_V2')}")
logger.info(f"LangSmith Project: {os.environ.get('LANGCHAIN_PROJECT')}")

#============================= create tools =========================================
@tool
def calculator(operator: Literal["add", "multiply", "divide", "subtract"], num1: int, num2: int) -> dict:
    """Calculate any arithmetic operation like add, divide, subtract and multiply"""
    try:
        if operator == "add":
            result = num1 + num2
        elif operator == "multiply":
            result = num1 * num2
        elif operator == "subtract":
            result =  num1 - num2
        elif operator == "divide":
            if num2 == 0:
                return {"error": "Division by zero is not allowed!"}
            result = num1/num2
        else:
            return {"error": f"Unsupported Operation {operator}"}

        return {"num1": num1, "num2": num2, "operation": operator, "result": result}
    except Exception as e:
        return {"error": str(e)}
    
search_tool = DuckDuckGoSearchRun()

def document_retriver_tool(document_path, name, desc):
    doc_loader = PyPDFLoader(document_path)
    documents = doc_loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=['\n']
    )
    docs = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever()
    retriever_tool = create_retriever_tool(retriever=retriever, name=name,
                                       description=desc)
    return retriever_tool

name = "Java Language"
desc = "search for information about Java programming Language. For any question about Java, use this tool"
document_path = "Java_Full_Notes.pdf"
retriever_tool = document_retriver_tool(document_path=document_path, name=name, desc=desc)
logger.info("Tools Loaded SuccessFully")

# ========================= Initialize LLM with tools ===================================
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)
tools = [calculator, search_tool, retriever_tool]
llm_with_tools = llm.bind_tools(tools)

# ===================== create connection in sqlite and checkpoint ==========================
conn = sqlite3.connect('chatbot.db', check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# ============================== create workflow =========================================
class ChatState(TypedDict):
    """State definition for the chat workflow."""
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState) -> dict:
    """
    Process chat messages through the LLM.
    
    Args:
        state: Current chat state containing message history
        
    Returns:
        Dictionary with the LLM's response message
    """
    try:
        output = llm_with_tools.invoke(state["messages"])
        return {"messages": [output]}
    except Exception as e:
        logger.error(f"Error in chat_node: {str(e)}")
        raise

tool_node = ToolNode(tools)

def load_workflow():
    """
    Create and compile the LangGraph workflow with memory persistence.
    
    Returns:
        Compiled workflow ready for execution
    """
    try:
        # Create state graph
        graph = StateGraph(ChatState)

        # Add nodes and edges
        graph.add_node("chat_node", chat_node)
        graph.add_node("tools", tool_node)

        graph.add_edge(START, "chat_node")
        graph.add_conditional_edges("chat_node", tools_condition)
        graph.add_edge("tools", "chat_node")

        workflow = graph.compile(checkpointer=checkpointer)

        logger.info("Workflow loaded successfully")
        return workflow
    
    except Exception as e:
        logger.error(f"Error loading workflow: {str(e)}")
        raise

# ================= SOME UTILS FUNCTION FOR FRONTEND =======================================
def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id']) # type: ignore

    return list(all_threads)

def delete_thread(thread_id):
    """
    Delete a conversation thread from the database.
    
    Args:
        thread_id: The thread ID to delete
    """
    try:
        cursor = conn.cursor()
        # Delete all checkpoints for this thread_id
        cursor.execute(
            "DELETE FROM checkpoints WHERE thread_id = ?",
            (str(thread_id),)
        )
        conn.commit()
        logger.info(f"Thread {thread_id} deleted successfully")
        return True
    except Exception as e:
        logger.error(f"Error deleting thread {thread_id}: {str(e)}")
        conn.rollback()
        return False

