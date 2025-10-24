from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv
import logging
import sqlite3

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

# create connection in sqlite
conn = sqlite3.connect('chatbot.db', check_same_thread=False)
# Setup checkpointer for conversation memory
checkpointer = SqliteSaver(conn=conn)

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
        output = llm.invoke(state["messages"])
        return {"messages": [output]}
    except Exception as e:
        logger.error(f"Error in chat_node: {str(e)}")
        raise


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
        graph.add_edge(START, "chat_node")
        graph.add_edge("chat_node", END)

        workflow = graph.compile(checkpointer=checkpointer)

        logger.info("Workflow loaded successfully")
        return workflow
    
    except Exception as e:
        logger.error(f"Error loading workflow: {str(e)}")
        raise

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

