import streamlit as st
from utils import load_workflow
from langchain_core.messages import HumanMessage


# ========================= state management =====================
if "workflow" not in st.session_state:
    st.session_state.workflow = load_workflow()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = 0

if "thread_list" not in st.session_state:
    st.session_state.thread_list = [st.session_state.thread_id]

# ======================== sidebar code ========================
def new_chat():
    thread_id = st.session_state.thread_id
    st.session_state.thread_id = thread_id + 1
    st.session_state.chat_history = []
    st.session_state.thread_list.append(st.session_state.thread_id)

def change_thread(thread_id):
    st.session_state.thread_id = thread_id

    state = st.session_state.workflow.get_state({"configurable": {"thread_id": thread_id}})
    
    # Extract messages from the state and convert to chat history format
    if state and hasattr(state, 'values') and 'messages' in state.values:
        messages = state.values['messages']
        st.session_state.chat_history = []
        
        for msg in messages:
            if hasattr(msg, 'type'):
                role = "user" if msg.type == "human" else "ai"
                st.session_state.chat_history.append({"role": role, "chat": msg.content})
    else:
        st.session_state.chat_history = []

st.sidebar.title("Conversation History")
if st.sidebar.button("➕ New Chat"):
    new_chat()

st.sidebar.divider()

for thread_id in st.session_state.thread_list:
    if st.sidebar.button(f"thread id: {thread_id}", key=f"btn_thread_{thread_id}"):
        change_thread(thread_id)
        st.rerun()

# ========================== UI ==================================
st.title(f"Simple ChatGpt - Thread ID {st.session_state.thread_id}")

for chat_history in st.session_state.chat_history:
    with st.chat_message(chat_history["role"]):
        st.text(chat_history["chat"])

user_input = st.chat_input("Enter your text here ... ")

CONFIG = {"configurable": {"thread_id": str(st.session_state.thread_id)}}

if user_input:
    with st.chat_message("human"):
        st.text(user_input)
        st.session_state.chat_history.append({"role": "user", "chat": user_input})

    with st.chat_message("ai"):
        with st.spinner("thinking..."):
            response = st.write_stream(
                message_chunk for message_chunk, meta_data in st.session_state.workflow.stream({"messages": [HumanMessage(content=user_input)]}, config=CONFIG, stream_mode="messages")
            ) # type: ignore

            st.session_state.chat_history.append({"role": "ai", "chat": response})
    