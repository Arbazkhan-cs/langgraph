import streamlit as st
import os
import json
from datetime import datetime
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils import get_youtube_transcript, create_retriever, format_docs, extract_video_id
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="YouTube AI Assistant",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling
st.markdown("""
<style>
    /* Global styles */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Header */
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    
    .sub-header {
        text-align: center;
        color: #6b7280;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Card containers */
    .video-card {
        background: white;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 20px;
        border: 1px solid #e5e7eb;
    }
    
    .chat-card {
        background: white;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 20px;
        border: 1px solid #e5e7eb;
    }
    
    .input-card {
        background: white;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 20px;
        border: 1px solid #e5e7eb;
        position: sticky;
        top: 20px;
        z-index: 100;
    }
    
    /* Chat messages */
    .question-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 16px 20px;
        border-radius: 12px 12px 12px 4px;
        margin: 12px 0;
        box-shadow: 0 2px 4px rgba(102, 126, 234, 0.2);
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    .question-box strong {
        display: block;
        margin-bottom: 6px;
        font-size: 0.85rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .answer-box {
        background: #f9fafb;
        color: #1f2937;
        padding: 16px 20px;
        border-radius: 12px 12px 4px 12px;
        margin: 12px 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        font-size: 0.95rem;
        line-height: 1.7;
    }
    
    .answer-box strong {
        display: block;
        margin-bottom: 8px;
        color: #667eea;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Suggested questions */
    .suggested-question {
        background: #f3f4f6;
        padding: 10px 16px;
        border-radius: 20px;
        margin: 6px 4px;
        display: inline-block;
        cursor: pointer;
        border: 1px solid #e5e7eb;
        transition: all 0.2s ease;
        font-size: 0.9rem;
        color: #4b5563;
    }
    
    .suggested-question:hover {
        background: #667eea;
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
    }
    
    /* API Status badges */
    .api-status {
        padding: 12px 16px;
        border-radius: 10px;
        margin: 8px 0;
        font-size: 0.9em;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .api-status.success {
        background-color: #d1fae5;
        color: #065f46;
        border: 1px solid #6ee7b7;
    }
    
    .api-status.error {
        background-color: #fee2e2;
        color: #991b1b;
        border: 1px solid #fca5a5;
    }
    
    .api-status.info {
        background-color: #dbeafe;
        color: #1e40af;
        border: 1px solid #93c5fd;
    }
    
    /* Progress indicators */
    .progress-item {
        background: #f9fafb;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 8px 0;
        border-left: 3px solid #667eea;
        font-size: 0.9rem;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        border: none;
        padding: 12px 28px;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px -1px rgba(102, 126, 234, 0.3);
        letter-spacing: 0.3px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px -1px rgba(102, 126, 234, 0.4);
    }
    
    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left: 4px solid #f59e0b;
        padding: 16px 20px;
        border-radius: 8px;
        margin: 16px 0;
        color: #78350f;
        font-size: 0.95rem;
    }
    
    /* Video info section */
    .video-info {
        background: #f9fafb;
        padding: 16px;
        border-radius: 10px;
        border: 1px solid #e5e7eb;
        margin-top: 12px;
    }
    
    .video-info-item {
        display: flex;
        justify-content: space-between;
        padding: 8px 0;
        border-bottom: 1px solid #e5e7eb;
    }
    
    .video-info-item:last-child {
        border-bottom: none;
    }
    
    .video-info-label {
        color: #6b7280;
        font-weight: 500;
        font-size: 0.9rem;
    }
    
    .video-info-value {
        color: #1f2937;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 60px 20px;
        color: #6b7280;
    }
    
    .empty-state-icon {
        font-size: 4rem;
        margin-bottom: 16px;
        opacity: 0.5;
    }
    
    .empty-state-text {
        font-size: 1.1rem;
        margin-bottom: 8px;
        color: #374151;
        font-weight: 500;
    }
    
    .empty-state-subtext {
        font-size: 0.95rem;
        color: #9ca3af;
    }
    
    /* Input fields */
    .stTextInput input, .stTextArea textarea {
        border-radius: 10px;
        border: 2px solid #e5e7eb;
        padding: 12px 16px;
        font-size: 0.95rem;
        transition: all 0.2s ease;
    }
    
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Divider */
    hr {
        margin: 24px 0;
        border: none;
        border-top: 2px solid #e5e7eb;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 24px;
        color: #6b7280;
        font-size: 0.9rem;
        margin-top: 40px;
        border-top: 2px solid #e5e7eb;
    }
    
    .footer a {
        color: #667eea;
        text-decoration: none;
        font-weight: 600;
    }
    
    .footer a:hover {
        text-decoration: underline;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #f9fafb;
        border-radius: 10px;
        font-weight: 600;
    }
    
    /* Chat history container */
    .chat-history {
        max-height: 500px;
        overflow-y: auto;
        padding-right: 10px;
    }
    
    /* Scrollbar */
    .chat-history::-webkit-scrollbar {
        width: 8px;
    }
    
    .chat-history::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    .chat-history::-webkit-scrollbar-thumb {
        background: #667eea;
        border-radius: 10px;
    }
    
    .chat-history::-webkit-scrollbar-thumb:hover {
        background: #764ba2;
    }
    /* Chat history scrollable container */
    .chat-history-container {
        scrollbar-width: thin;
        scrollbar-color: #667eea #f1f1f1;
    }
    
    .chat-history-container::-webkit-scrollbar {
        width: 10px;
    }
    
    .chat-history-container::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    .chat-history-container::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        border: 2px solid #f1f5f9;
    }
    
    .chat-history-container::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Smooth scrolling */
    .chat-history-container {
        scroll-behavior: smooth;
    }
    
    /* Add some spacing for better readability in scrollable area */
    .chat-history-container .question-box,
    .chat-history-container .answer-box {
        margin: 16px 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models(groq_api_key, hf_token=None):
    """Load and cache models with provided API keys"""
    try:
        os.environ["GROQ_API_KEY"] = groq_api_key
        if hf_token:
            os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
        
        model = ChatGroq(
            model="llama-3.3-70b-versatile", 
            temperature=0.5,
            groq_api_key=groq_api_key
        )
        
        embedding = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
        parser = StrOutputParser()
        
        prompt = PromptTemplate(
            template="""
            You are a helpful AI assistant specialized in answering questions about YouTube video content.
            Answer ONLY based on the provided transcript context from the video.
            If the context doesn't contain enough information to answer the question, politely say you don't have enough information from the video transcript.
            
            Be conversational, helpful, and provide detailed explanations when possible.
            If you can reference specific parts or timestamps from the video, that would be helpful.

            <context>
            {context}
            </context>

            Question: {question}
            
            Answer:
            """,
            input_variables=['context', 'question']
        )
        
        return model, embedding, parser, prompt, None
    except Exception as e:
        return None, None, None, None, str(e)

def validate_api_keys(groq_key, hf_token=None):
    """Validate API keys by testing model initialization"""
    try:
        test_model = ChatGroq(
            model="llama3-8b-8192",
            groq_api_key=groq_key,
            temperature=0.1
        )
        test_model.invoke("test")
        groq_valid = True
        groq_error = None
    except Exception as e:
        groq_valid = False
        groq_error = str(e)
    
    hf_valid = True
    hf_error = None
    
    return groq_valid, groq_error, hf_valid, hf_error

def create_qa_chain(url, model, embedding, parser, prompt):
    """Create the Q&A chain for a specific video"""
    with st.spinner("🎬 Extracting video transcript..."):
        result = get_youtube_transcript(url)
        
        if "error" in result:
            st.error(f"❌ {result['error']}")
            return None, None
        
        st.success(f"✅ Transcript extracted successfully! Found {result['total_segments']} segments.")
    
    with st.spinner("📄 Creating document chunks..."):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        document_chunks = text_splitter.create_documents([result['transcript_text']])
    
    with st.spinner("🔍 Creating vector index..."):
        retriever = create_retriever(document_chunks, embedding)
    
    with st.spinner("🔗 Building Q&A chain..."):
        parallel_chain = RunnableParallel({
            'context': retriever | RunnableLambda(format_docs),
            'question': RunnablePassthrough()
        })
        final_chain = parallel_chain | prompt | model | parser
    
    return final_chain, result

def save_recent_video(url, video_id):
    """Save video to recent history"""
    if 'recent_videos' not in st.session_state:
        st.session_state.recent_videos = []
    
    # Remove if already exists
    st.session_state.recent_videos = [v for v in st.session_state.recent_videos if v['url'] != url]
    
    # Add to beginning
    st.session_state.recent_videos.insert(0, {
        'url': url,
        'video_id': video_id,
        'timestamp': datetime.now().isoformat()
    })
    
    # Keep only last 5
    st.session_state.recent_videos = st.session_state.recent_videos[:5]

def export_chat_history():
    """Export chat history as formatted text"""
    if not st.session_state.chat_history:
        return None
    
    export_text = f"YouTube AI Assistant - Chat Export\n"
    export_text += f"Video: {st.session_state.current_url}\n"
    export_text += f"Video ID: {st.session_state.video_info['video_id']}\n"
    export_text += f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    export_text += "=" * 80 + "\n\n"
    
    for i, (question, answer) in enumerate(st.session_state.chat_history, 1):
        export_text += f"Q{i}: {question}\n\n"
        export_text += f"A{i}: {answer}\n\n"
        export_text += "-" * 80 + "\n\n"
    
    return export_text

def export_chat_history_json():
    """Export chat history as JSON"""
    if not st.session_state.chat_history:
        return None
    
    export_data = {
        'video_url': st.session_state.current_url,
        'video_id': st.session_state.video_info['video_id'],
        'exported_at': datetime.now().isoformat(),
        'chat_history': [
            {'question': q, 'answer': a} 
            for q, a in st.session_state.chat_history
        ]
    }
    
    return json.dumps(export_data, indent=2)

def get_suggested_questions():
    """Generate suggested questions based on video context"""
    return [
        "What is the main topic of this video?",
        "Can you summarize the key points discussed?",
        "What are the main conclusions or takeaways?",
        "Are there any important examples mentioned?",
        "What recommendations or advice are given?"
    ]

def main():
    # Header
    st.markdown('<h1 class="main-header">🎯 YouTube AI Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Analyze and chat with any YouTube video using AI</p>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None
    if 'video_info' not in st.session_state:
        st.session_state.video_info = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_url' not in st.session_state:
        st.session_state.current_url = ""
    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded = False
    if 'api_keys_valid' not in st.session_state:
        st.session_state.api_keys_valid = False
    if 'recent_videos' not in st.session_state:
        st.session_state.recent_videos = []
    if 'suggested_question_clicked' not in st.session_state:
        st.session_state.suggested_question_clicked = None
    
    # Sidebar for configuration
    with st.sidebar:
        st.markdown('<div class="section-header">⚙️ Configuration</div>', unsafe_allow_html=True)
        
        # API Keys section
        with st.expander("🔑 API Keys", expanded=not st.session_state.api_keys_valid):
            
            groq_api_key = st.text_input(
                "Groq API Key *",
                type="password",
                value=os.getenv("GROQ_API_KEY", ""),
                help="Required: Get your free API key from https://console.groq.com/",
                placeholder="gsk_..."
            )
            
            hf_token = st.text_input(
                "HuggingFace Token",
                type="password",
                value=os.getenv("HUGGINGFACE_HUB_TOKEN", ""),
                help="Optional: For better model access from https://huggingface.co/settings/tokens",
                placeholder="hf_..."
            )
            
            if st.button("🔍 Validate Keys", use_container_width=True):
                if groq_api_key:
                    with st.spinner("Validating..."):
                        groq_valid, groq_error, hf_valid, hf_error = validate_api_keys(groq_api_key, hf_token)
                        
                        if groq_valid:
                            st.markdown('<div class="api-status success">✅ Groq API Key: Valid</div>', unsafe_allow_html=True)
                            st.session_state.api_keys_valid = True
                        else:
                            st.markdown(f'<div class="api-status error">❌ Groq API Key: Invalid</div>', unsafe_allow_html=True)
                            st.session_state.api_keys_valid = False
                        
                        if hf_token:
                            st.markdown('<div class="api-status success">✅ HuggingFace Token: Provided</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="api-status info">ℹ️ HuggingFace Token: Optional</div>', unsafe_allow_html=True)
                else:
                    st.warning("⚠️ Please enter your Groq API key")
        
        st.markdown("---")
        
        # Recent Videos section
        if st.session_state.recent_videos:
            st.markdown('<div class="section-header">📜 Recent Videos</div>', unsafe_allow_html=True)
            
            recent_options = ["Select a recent video..."] + [
                f"📹 {v['video_id']} ({v['timestamp'][:10]})" 
                for v in st.session_state.recent_videos
            ]
            
            selected_recent = st.selectbox(
                "Quick Access",
                recent_options,
                label_visibility="collapsed"
            )
            
            if selected_recent != "Select a recent video...":
                idx = recent_options.index(selected_recent) - 1
                selected_video = st.session_state.recent_videos[idx]
                
                if st.button("🔄 Load This Video", use_container_width=True):
                    st.session_state.current_url = selected_video['url']
                    st.rerun()
            
            st.markdown("---")
        
        # Video Processing section
        st.markdown('<div class="section-header">🎬 Video Setup</div>', unsafe_allow_html=True)
        
        youtube_url = st.text_input(
            "YouTube URL",
            value=st.session_state.current_url if st.session_state.current_url else "",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Paste any YouTube video URL here"
        )
        
        if st.button("🚀 Process Video", type="primary", use_container_width=True):
            if not groq_api_key:
                st.error("❌ Please provide your Groq API key first!")
            elif not youtube_url:
                st.warning("⚠️ Please enter a YouTube URL")
            else:
                model, embedding, parser, prompt, error = load_models(groq_api_key, hf_token)
                
                if error:
                    st.error(f"❌ Error loading models: {error}")
                elif model is not None:
                    qa_chain, video_info = create_qa_chain(youtube_url, model, embedding, parser, prompt)
                    
                    if qa_chain is not None:
                        st.session_state.qa_chain = qa_chain
                        st.session_state.video_info = video_info
                        st.session_state.current_url = youtube_url
                        st.session_state.models_loaded = True
                        st.session_state.chat_history = []
                        
                        # Save to recent videos
                        video_id = extract_video_id(youtube_url)
                        if video_id:
                            save_recent_video(youtube_url, video_id)
                        
                        st.success("🎉 Ready to chat!")
                        st.rerun()
        
        # Video info
        if st.session_state.video_info:
            st.markdown("---")
            st.markdown('<div class="section-header">📊 Video Info</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="video-info">
                <div class="video-info-item">
                    <span class="video-info-label">Video ID</span>
                    <span class="video-info-value">{st.session_state.video_info['video_id']}</span>
                </div>
                <div class="video-info-item">
                    <span class="video-info-label">Segments</span>
                    <span class="video-info-value">{st.session_state.video_info['total_segments']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Export and Clear buttons
        if st.session_state.chat_history:
            st.markdown("---")
            st.markdown('<div class="section-header">💾 Chat Actions</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                export_text = export_chat_history()
                if export_text:
                    st.download_button(
                        "📥 TXT",
                        export_text,
                        f"chat_{st.session_state.video_info['video_id']}.txt",
                        "text/plain",
                        use_container_width=True
                    )
            
            with col2:
                export_json = export_chat_history_json()
                if export_json:
                    st.download_button(
                        "📥 JSON",
                        export_json,
                        f"chat_{st.session_state.video_info['video_id']}.json",
                        "application/json",
                        use_container_width=True
                    )
            
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        
        # Help section
        st.markdown("---")
        with st.expander("💡 Quick Tips"):
            st.markdown("""
            **Getting Started:**
            1. Add your Groq API key
            2. Paste a YouTube URL
            3. Click "Process Video"
            4. Start asking questions!
            
            **Features:**
            - 📜 Recent videos saved
            - 💾 Export chat history
            - 💡 Suggested questions
            - 🔄 Quick video switching
            """)
    
    # Main content area
    col1, col2 = st.columns([1, 1], gap="large")
    
    # Left column - Video player
    with col1:
        st.markdown('<div class="section-header">🎬 Video Player</div>', unsafe_allow_html=True)
        
        if youtube_url and extract_video_id(youtube_url):
            video_id = extract_video_id(youtube_url)
            st.video(f"https://www.youtube.com/watch?v={video_id}")
        else:
            st.markdown("""
            <div class="empty-state">
                <div class="empty-state-icon">📹</div>
                <div class="empty-state-text">No Video Loaded</div>
                <div class="empty-state-subtext">Enter a YouTube URL in the sidebar to get started</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Right column - Chat interface
    with col2:
        # Question input at top
        st.markdown('<div class="section-header">💬 Ask a Question</div>', unsafe_allow_html=True)
        
        if st.session_state.qa_chain is not None:
            # Use text_area for better input experience
            user_question = st.text_area(
                "Your question",
                value=st.session_state.suggested_question_clicked if st.session_state.suggested_question_clicked else "",
                placeholder="What is this video about? What are the key takeaways?",
                height=80,
                key="user_question",
                label_visibility="collapsed"
            )
            
            col_ask, col_clear = st.columns([3, 1])
            with col_ask:
                ask_button = st.button("💡 Ask Question", type="primary", use_container_width=True)
            
            # Process question
            if ask_button:
                if user_question.strip():
                    with st.spinner("🧠 Thinking..."):
                        try:
                            answer = st.session_state.qa_chain.invoke(user_question)
                            st.session_state.chat_history.append((user_question, answer))
                            st.session_state.suggested_question_clicked = None
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error generating answer: {str(e)}")
                else:
                    st.warning("⚠️ Please enter a question!")
        else:
            st.markdown("""
            <div class="info-box">
                ⚡ <strong>Quick Start:</strong> Configure your API keys and process a video to begin chatting!
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Chat history below input
    st.markdown('<div class="section-header">💭 Chat History</div>', unsafe_allow_html=True)
    
    if st.session_state.chat_history:
        # Reverse the chat history to show latest at the top
        for i, (question, answer) in enumerate(reversed(st.session_state.chat_history)):
            # Calculate the actual question number (reversed index)
            question_number = len(st.session_state.chat_history) - i
            st.markdown(f"""
            <div class="question-box">
                <strong>Question #{question_number}</strong>
                <div style="margin-top: 5px;">{question}</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class="answer-box">
                <strong>AI Response</strong>
                <div style="margin-top: 5px;">{answer}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        if st.session_state.qa_chain is not None:
            st.markdown("""
            <div class="empty-state">
                <div class="empty-state-icon">💭</div>
                <div class="empty-state-text">No messages yet</div>
                <div class="empty-state-subtext">Ask your first question above to start the conversation</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="empty-state">
                <div class="empty-state-icon">🎯</div>
                <div class="empty-state-text">Ready to Get Started?</div>
                <div class="empty-state-subtext">Process a video to begin asking questions</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>🚀 Powered by <strong>Streamlit</strong>, <strong>LangChain</strong>, and <strong>Groq</strong></p>
        <p>🔑 Get your free API key at <a href="https://console.groq.com/" target="_blank">console.groq.com</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()