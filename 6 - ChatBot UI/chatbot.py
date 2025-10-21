from utils import load_workflow
import streamlit as st
from langchain_core.messages import HumanMessage

# Set page configuration
st.set_page_config(page_title="ChatBot", page_icon="🤖", layout="wide")

# Initialize session state variables
if "workflow" not in st.session_state:
    st.session_state.workflow = load_workflow()

if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {1: []}  # Dictionary: {thread_id: messages}

if "current_thread_id" not in st.session_state:
    st.session_state.current_thread_id = 1

if "next_thread_id" not in st.session_state:
    st.session_state.next_thread_id = 2

# Sidebar for chat management
with st.sidebar:
    st.title("💬 Chat Sessions")
    
    # New Chat button
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        new_thread_id = st.session_state.next_thread_id
        st.session_state.chat_sessions[new_thread_id] = []
        st.session_state.current_thread_id = new_thread_id
        st.session_state.next_thread_id += 1
        st.rerun()
    
    st.divider()
    
    # Display all chat sessions
    st.subheader("Your Chats")
    
    # Sort thread IDs in descending order (newest first)
    sorted_thread_ids = sorted(st.session_state.chat_sessions.keys(), reverse=True)
    
    for thread_id in sorted_thread_ids:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Get first message as chat preview (or show "Empty chat")
            messages = st.session_state.chat_sessions[thread_id]
            if messages:
                preview = messages[0][:30] + "..." if len(messages[0]) > 30 else messages[0]
            else:
                preview = "Empty chat"
            
            # Highlight current chat
            if thread_id == st.session_state.current_thread_id:
                button_label = f"🟢 Chat {thread_id}: {preview}"
                button_type = "primary"
            else:
                button_label = f"Chat {thread_id}: {preview}"
                button_type = "secondary"
            
            if st.button(button_label, key=f"switch_{thread_id}", use_container_width=True, type=button_type):
                st.session_state.current_thread_id = thread_id
                st.rerun()
        
        with col2:
            # Delete button
            if st.button("🗑️", key=f"delete_{thread_id}", help=f"Delete Chat {thread_id}"):
                # Don't allow deletion if it's the only chat
                if len(st.session_state.chat_sessions) > 1:
                    del st.session_state.chat_sessions[thread_id]
                    
                    # If deleted current chat, switch to another
                    if thread_id == st.session_state.current_thread_id:
                        remaining_threads = list(st.session_state.chat_sessions.keys())
                        st.session_state.current_thread_id = remaining_threads[0]
                    
                    st.rerun()
                else:
                    st.warning("Cannot delete the last chat!")
    
    st.divider()
    
    # Chat statistics
    st.caption(f"📊 Total Chats: {len(st.session_state.chat_sessions)}")
    st.caption(f"🆔 Current Thread: {st.session_state.current_thread_id}")

# Main chat interface
st.title(f"🤖 ChatBot - Thread {st.session_state.current_thread_id}")

# Get current thread's messages
current_messages = st.session_state.chat_sessions[st.session_state.current_thread_id]

# Display chat history
chat_container = st.container()
with chat_container:
    if not current_messages:
        st.info("👋 Start a new conversation! Type your message below.")
    else:
        # Display messages in pairs (user and assistant)
        for i in range(0, len(current_messages), 2):
            # User message
            with st.chat_message("user"):
                st.markdown(current_messages[i])
            
            # Assistant message (if exists)
            if i + 1 < len(current_messages):
                with st.chat_message("assistant"):
                    st.markdown(current_messages[i + 1])

# Chat input
if prompt := st.chat_input("Type your message here...", key=f"input_{st.session_state.current_thread_id}"):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add user message to current thread's history
    current_messages.append(prompt)
    
    # Create configuration with thread_id for persistence
    config = {"configurable": {"thread_id": str(st.session_state.current_thread_id)}}
    
    # Get response from workflow
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Invoke workflow with the user's message
                response = st.session_state.workflow.invoke(
                    {"messages": [HumanMessage(content=prompt)]},
                    config=config # type: ignore
                )
                
                # Extract assistant's response
                assistant_message = response["messages"][-1].content
                st.markdown(assistant_message)
                
                # Add assistant message to current thread's history
                current_messages.append(assistant_message)
                
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                current_messages.append(error_msg)
    
    # Update the session state
    st.session_state.chat_sessions[st.session_state.current_thread_id] = current_messages

# Footer
st.divider()
st.caption("Powered by LangGraph & Groq 🚀")
