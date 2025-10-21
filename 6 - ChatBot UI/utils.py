from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7, max_tokens=200)


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

        # Setup checkpointer for conversation memory
        checkpointer = InMemorySaver()
        workflow = graph.compile(checkpointer=checkpointer)

        logger.info("Workflow loaded successfully")
        return workflow
    
    except Exception as e:
        logger.error(f"Error loading workflow: {str(e)}")
        raise
