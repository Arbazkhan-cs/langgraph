import re
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.vectorstores import FAISS
from youtube_transcript_api.proxies import WebshareProxyConfig


def extract_video_id(url):
    """
    Extract video ID from various YouTube URL formats
    
    Args:
        url (str): YouTube video URL
        
    Returns:
        str or None: Video ID if found, None otherwise
    """
    if not url:
        return None
        
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
        r'youtube\.com\/watch\?.*v=([^&\n?#]+)',
        r'youtube\.com\/v\/([^&\n?#]+)',
        r'youtube\.com\/shorts\/([^&\n?#]+)'
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_youtube_transcript(youtube_url):
    """
    Extract transcript from a YouTube video URL

    Args:
        youtube_url (str): YouTube video URL

    Returns:
        dict: Contains transcript text, raw transcript data, and metadata
              or error message if extraction fails
    """
    try:
        # Extract video ID from URL
        video_id = extract_video_id(youtube_url)
        if not video_id:
            return {"error": "Invalid YouTube URL format"}

        # Try to get transcript in preferred languages
        transcript_list = None
        languages_to_try = ['en', 'en-US', 'en-GB', 'en-CA', 'en-AU']
        ytt_api = YouTubeTranscriptApi(
            proxy_config=WebshareProxyConfig(
                proxy_username="rmyfszdc",
                proxy_password="fd01a2fscmhn"
            )
        )
        
        try:
            # First try English transcripts
            for lang in languages_to_try:
                try:
                    transcript_list = ytt_api.fetch(video_id, languages=[lang])
                    break
                except:
                    continue
            
            # If no English transcript found, try any available transcript
            if transcript_list is None:
                available_transcripts = ytt_api.list(video_id)
                # Get the first available transcript
                for transcript in available_transcripts:
                    try:
                        transcript_list = transcript.fetch()
                        break
                    except:
                        continue
                        
        except Exception as e:
            return {"error": f"No transcript available for this video: {str(e)}"}
        
        if transcript_list is None:
            return {"error": "No transcript could be retrieved for this video"}

        # Extract text from transcript
        text_transcript = " ".join(f"[start: {transcript.start}] {transcript.text}" for transcript in transcript_list)
        
        # Clean up the transcript text
        text_transcript = clean_transcript_text(text_transcript)

        return {
            "video_id": video_id,
            "transcript_text": text_transcript,
            "total_segments": len(transcript_list)
        }

    except Exception as e:
        return {"error": f"Failed to get transcript: {str(e)}"}

def clean_transcript_text(text):
    """
    Clean and format transcript text
    
    Args:
        text (str): Raw transcript text
        
    Returns:
        str: Cleaned transcript text
    """
    if not text:
        return ""
    
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text)
    
    # Remove common transcript artifacts
    text = re.sub(r'\[.*?\]', '', text)  # Remove [Music], [Applause], etc.
    text = re.sub(r'\(.*?\)', '', text)  # Remove parenthetical content
    
    # Fix common issues
    text = text.replace('  ', ' ')  # Remove double spaces
    text = text.strip()  # Remove leading/trailing whitespace
    
    return text

def create_retriever(document_chunks, embedding, search_kwargs=None):
    """
    Create vectorstore from document chunks and return a retriever
    
    Args:
        document_chunks (list): List of document chunks
        embedding: Embedding model instance
        search_kwargs (dict): Optional search parameters
        
    Returns:
        VectorStoreRetriever: Configured retriever instance
    """
    if not document_chunks:
        raise ValueError("Document chunks cannot be empty")
    
    if search_kwargs is None:
        search_kwargs = {'k': 5}
    
    try:
        # Create FAISS vectorstore
        vectorstore = FAISS.from_documents(
            documents=document_chunks,
            embedding=embedding
        )
        
        # Return retriever with specified search parameters
        return vectorstore.as_retriever(
            search_type="similarity", 
            search_kwargs=search_kwargs
        )
        
    except Exception as e:
        raise Exception(f"Failed to create retriever: {str(e)}")

def format_docs(retrieved_docs):
    """
    Format retrieved documents for context
    
    Args:
        retrieved_docs (list): List of retrieved document objects
        
    Returns:
        str: Formatted context text
    """
    if not retrieved_docs:
        return "No relevant context found."
    
    # Join document contents with clear separation
    context_parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        content = doc.page_content.strip()
        if content:
            context_parts.append(f"[Context {i}]: {content}")
    
    return "\n\n".join(context_parts) if context_parts else "No relevant context found."

def validate_youtube_url(url):
    """
    Validate if a given URL is a valid YouTube URL
    
    Args:
        url (str): URL to validate
        
    Returns:
        tuple: (is_valid, video_id_or_error_message)
    """
    if not url or not isinstance(url, str):
        return False, "URL must be a non-empty string"
    
    video_id = extract_video_id(url)
    if video_id:
        return True, video_id
    else:
        return False, "Invalid YouTube URL format"

# Utility function for testing
def test_transcript_extraction(youtube_url):
    """
    Test function to check transcript extraction
    
    Args:
        youtube_url (str): YouTube URL to test
        
    Returns:
        dict: Test results
    """
    print(f"Testing transcript extraction for: {youtube_url}")
    
    # Validate URL
    is_valid, result = validate_youtube_url(youtube_url)
    if not is_valid:
        return {"error": f"Invalid URL: {result}"}
    
    video_id = result
    print(f"Video ID: {video_id}")
    
    # Extract transcript
    transcript_result = get_youtube_transcript(youtube_url)
    
    if "error" in transcript_result:
        print(f"Error: {transcript_result['error']}")
        return transcript_result
    else:
        print(f"Success! Extracted {transcript_result['total_segments']} transcript segments")
        print(f"First 200 characters: {transcript_result['transcript_text'][:200]}...")
        return transcript_result

# Example usage and testing
if __name__ == "__main__":
    # Test with a sample YouTube URL
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    result = test_transcript_extraction(test_url)
    
    if "error" not in result:
        print("\n" + "="*50)
        print("TRANSCRIPT EXTRACTION SUCCESSFUL")
        print("="*50)
        print(f"Video ID: {result['video_id']}")
        print(f"Total segments: {result['total_segments']}")
        print(f"Transcript length: {len(result['transcript_text'])} characters")
    else:
        print(f"\nERROR: {result['error']}")