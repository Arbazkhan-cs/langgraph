import streamlit as st
from utility_for_sqlite_integration import load_workflow, retrieve_all_threads, delete_thread
from langchain_core.messages import HumanMessage, AIMessage
import uuid

# **************************************** utility functions *************************

def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_conversation(thread_id):
    state = st.session_state['workflow'].get_state(config={'configurable': {'thread_id': thread_id}})
    # Check if messages key exists in state values, return empty list if not
    return state.values.get('messages', [])


# **************************************** Session Setup ******************************
if 'workflow' not in st.session_state:
    st.session_state['workflow'] = load_workflow()

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()

add_thread(st.session_state['thread_id'])


# **************************************** Sidebar UI *********************************

st.sidebar.title('LangGraph Chatbot')

if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.header('My Conversations')

for thread_id in st.session_state['chat_threads'][::-1]:
    col1, col2 = st.sidebar.columns([4, 1])
    
    with col1:
        if st.button("conversation: "+str(thread_id)[:8], key=f"thread_btn_{thread_id}", use_container_width=True):
            st.session_state['thread_id'] = thread_id
            messages = load_conversation(thread_id)

            temp_messages = []

            for msg in messages:
                if isinstance(msg, HumanMessage):
                    role='user'
                else:
                    role='assistant'
                temp_messages.append({'role': role, 'content': msg.content})

            st.session_state['message_history'] = temp_messages
            st.rerun()
    
    with col2:
        if st.button("🗑️", key=f"delete_btn_{thread_id}", help="Delete this conversation"):
            if len(st.session_state['chat_threads']) > 1:
                if delete_thread(thread_id):
                    st.session_state['chat_threads'].remove(thread_id)
                    
                    # If deleted current chat, switch to another
                    if thread_id == st.session_state['thread_id']:
                        st.session_state['thread_id'] = st.session_state['chat_threads'][0]
                        st.session_state['message_history'] = []
                    
                    st.rerun()
                else:
                    st.sidebar.error("Failed to delete conversation")
            else:
                st.sidebar.warning("Cannot delete the last conversation!")


# **************************************** Main UI ************************************
st.title(f"Simple ChatGpt: {str(st.session_state['thread_id'])[:8]}")

# loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input('Type here')

if user_input:

    # first add the message to message_history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}

     # first add the message to message_history
    with st.chat_message("assistant"):
        with st.spinner():
            def ai_only_stream():
                for message_chunk, metadata in st.session_state["workflow"].stream(
                    {"messages": [HumanMessage(content=user_input)]}, # type: ignore
                    config=CONFIG,
                    stream_mode="messages"
                ):
                    if isinstance(message_chunk, AIMessage): 
                        # yield only assistant tokens
                        yield message_chunk.content

            ai_message = st.write_stream(ai_only_stream())

        st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
