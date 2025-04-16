import os
import json
import requests
import time
import logging
import streamlit as st
from dotenv import load_dotenv
from requests_aws4auth import AWS4Auth
from openai import OpenAI, RateLimitError
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from docx import Document
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
import boto3
from io import BytesIO
import pyrebase
import sys

load_dotenv()

api_key = os.getenv("apiKey")
authDomain = os.getenv("authDomain")
databaseURL = os.getenv("databaseURL")
projectId = os.getenv("projectId")
storageBucket = os.getenv("storageBucket")
messagingSenderId = os.getenv("messagingSenderId")
appId = os.getenv("appId")
measurementId = os.getenv("measurementId")

# ---------- CONFIGURATION ----------
# Firebase config
firebase_config = {
    "apiKey": api_key,
    "authDomain": authDomain,
    "databaseURL": databaseURL,
    "projectId":projectId ,
    "storageBucket": storageBucket,
    "messagingSenderId": messagingSenderId,
    "appId": appId,
    "measurementId": measurementId
}


# AWS credentials and region
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION")

# OpenAI API Key
OPENAI_API_KEY = os.getenv("CLIENT_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# OpenSearch Serverless endpoint
OPENSEARCH_ENDPOINT = os.getenv("OPENSEARCH_ENDPOINT")

# AWS Signature Version 4 authentication
awsauth = AWS4Auth(AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_REGION, "aoss")

# Define the index name
INDEX_NAME = "documents"

# Initialize Firebase
firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()
# Document chunking settings
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 500

# Configure logging
logging.basicConfig(level=logging.INFO)

# AWS Clients
translate_client = boto3.client("translate", region_name=AWS_REGION)
comprehend_client = boto3.client("comprehend", region_name=AWS_REGION)

# Function to detect language
def detect_language(text):
    response = comprehend_client.detect_dominant_language(Text=text)
    languages = response["Languages"]
    if languages:
        return languages[0]["LanguageCode"]
    return "unknown"

def handle_login(email, password):
    """Handle user login with Firebase."""
    try:
        user = auth.sign_in_with_email_and_password(email, password)
        st.session_state.user = user
        st.session_state.user_email = email
        st.success("Login successful!")
        st.rerun()
        return True
    except Exception as e:
        error_message = str(e)
        if "INVALID_EMAIL" in error_message:
            st.error("Invalid email address.")
        elif "INVALID_PASSWORD" in error_message:
            st.error("Incorrect password.")
        else:
            st.error(f"Login failed: {error_message}")
        return False


def handle_password_reset(email):
    """Send password reset email."""
    try:
        auth.send_password_reset_email(email)
        st.success(f"Password reset email sent to {email}")
    except Exception as e:
        error_message = str(e)
        if "INVALID_EMAIL" in error_message:
            st.error("This email is not registered.")
        else:
            st.error(f"Password reset failed: {error_message}")

def handle_logout():
    """Clear session state on logout."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.success("You have been logged out successfully.")

def split_text_by_bytes(text, max_bytes=9000):
    """Splits text into chunks ensuring each is within the max byte size."""
    chunks = []
    current_chunk = []
    current_size = 0

    for word in text.split():
        word_size = sys.getsizeof(word.encode('utf-8'))  # Get word size in bytes
        if current_size + word_size > max_bytes:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_size = word_size
        else:
            current_chunk.append(word)
            current_size += word_size

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def translate_to_english(text):
    """Translates text to English while handling AWS Translate's size limit."""
    source_language = detect_language(text[:1000])  # Detect language from a sample
    if source_language == "en":
        return text

    text_chunks = split_text_by_bytes(text, max_bytes=9000)  # Dynamically split text
    translated_chunks = []

    for chunk in text_chunks:
        response = translate_client.translate_text(
            Text=chunk,
            SourceLanguageCode=source_language,
            TargetLanguageCode="en"
        )
        translated_chunks.append(response["TranslatedText"])

    return " ".join(translated_chunks)


# Function to read PDF text
def read_pdf(file):
    file.seek(0)  # Reset file pointer
    pdf_reader = PdfReader(BytesIO(file.read()))  # Convert to BytesIO
    text = []
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text.append(page_text)
    return "\n".join(text) if text else "Could not extract text from PDF."

# Function to read DOCX text
def read_docx(file):
    file.seek(0)  # Reset file pointer
    doc = Document(BytesIO(file.read()))  # Convert to BytesIO
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])


# Function to split text into chunks
def chunk_document(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_text(text)

def show_auth_page():
    """Display authentication options (login/signup/forgot password)."""
    st.title(":bamboo: Bamboo Species Chatbot Login")
    st.caption("Access your account securely.")
        
    auth_option = st.radio("Choose an option:", 
                          ["Login"])
    
    if auth_option == "Login":
        with st.form("login_form"):
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            submit = st.form_submit_button("Login")
            
            if submit:
                handle_login(email, password)
    
   
    
    elif auth_option == "Forgot Password":
        with st.form("reset_form"):
            email = st.text_input("Email", key="reset_email")
            submit = st.form_submit_button("Send Reset Link")
            
            if submit:
                handle_password_reset(email)

# Function to handle OpenAI rate limits with retry
@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_random_exponential(min=1, max=20),
    stop=stop_after_attempt(5),
    before_sleep=lambda retry_state: logging.warning(f"Retrying due to rate limit... Attempt {retry_state.attempt_number}")
)
def generate_embeddings(text_chunks):
    try:
        response = openai_client.embeddings.create(
            input=text_chunks,
            model="text-embedding-ada-002"
        )
        return [res.embedding for res in response.data]
    except RateLimitError as e:
        logging.error(f"RateLimitError: {e}")
        time.sleep(10)
        raise

# Function to retrieve relevant documents
def retrieve_relevant_documents(query, top_k=10):
    query_embedding = generate_embeddings([query])[0]
    search_url = f"{OPENSEARCH_ENDPOINT}/{INDEX_NAME}/_search"
    headers = {"Content-Type": "application/json"}
    payload = {
        "size": top_k,
        "query": {
            "knn": {
                "embeddings": {
                    "vector": query_embedding,
                    "k": top_k
                }
            }
        },
        "_source": ["text"]
    }
    
    response = requests.get(search_url, auth=awsauth, headers=headers, json=payload)
    retrieved_docs = []
    if response.status_code == 200:
        results = response.json()
        for hit in results.get("hits", {}).get("hits", []):
            retrieved_docs.append(hit["_source"].get("text", ""))
    
    
    
    return retrieved_docs

# Function to generate an answer using retrieved document context
def generate_answer(query):
    retrieved_docs = retrieve_relevant_documents(query, top_k=5)
    
    context = "\n\n".join(retrieved_docs[:5]) if retrieved_docs else ""

    prompt = f"""
    Answer the query below in a clear and informative way. If relevant document context is available, use it to enhance your response. 
    If document context is insufficient or missing, use your general knowledge and research-based insights to provide a well-informed answer.
    
    DO NOT mention whether the document contains the answer or not—simply provide the best possible response.

    ### Document Context (if available):
    {context}

    ### Query:
    {query}

    ### Answer:
    """

    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an AI assistant that answers queries using both document context and general knowledge. Ensure responses are complete and useful without stating document limitations."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content


def show_qa_page():
    st.title(":bamboo: Bamboo Species Chatbot")
    query = st.text_input("Ask a question about the documents", placeholder="Enter your query here...")
    if st.button("Generate Answer"):
            if query.strip():
                with st.spinner("Generating answer..."):
                    answer = generate_answer(query)
                    st.write("📄 Generated Answer:")
                    st.write(answer)
            else:
                st.warning("Please enter a query before clicking the button.")
   

def main():
    st.set_page_config(page_title="Document QA", page_icon="📄", layout="wide")

    
    # Custom CSS for better styling
    st.markdown("""
    <style>
        .stTextInput input, .stTextArea textarea {
            border-radius: 8px !important;
        }
        .stButton button {
            border-radius: 8px !important;
            background-color: #4CAF50 !important;
            color: white !important;
        }
        .stAlert {
            border-radius: 8px !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if "user" not in st.session_state:
        st.session_state.user = None
    if "user_email" not in st.session_state:
        st.session_state.user_email = None
    
    # Show appropriate page based on auth state
    if st.session_state.user:
        # Navigation sidebar
        st.sidebar.title(f"Welcome, {st.session_state.user_email}")
        if st.sidebar.button("Logout"):
            handle_logout()
            st.rerun()
        
        app_mode = st.sidebar.radio(
            "Navigation",
            ["Ask Questions"],
            index=0
        )
        
        if app_mode == "Ask Questions":
            show_qa_page()
        
    else:
        show_auth_page()

if __name__ == "__main__":
    main()
# if user_choice == "Upload New Files":
#     uploaded_files = st.file_uploader("Upload PDFs or DOCX files", accept_multiple_files=True, type=["pdf", "docx"])
    
#     if uploaded_files:
#         for uploaded_file in uploaded_files:
#             with st.spinner(f"Processing {uploaded_file.name}..."):
#                 if uploaded_file.name.endswith(".pdf"):
#                     text = read_pdf(uploaded_file)
#                 elif uploaded_file.name.endswith(".docx"):
#                     text = read_docx(uploaded_file)
#                 else:
#                     st.error(f"Unsupported file type: {uploaded_file.name}")
#                     continue
                
#                 # Detect language and display to user
#                 source_language = detect_language(text[:1000])  # Using sample for faster detection
#                 st.info(f"Detected language for {uploaded_file.name}: {source_language}")
                
#                 if source_language != "en":
#                     st.info(f"Translating from {source_language} to English (target language)...")
#                     text = translate_to_english(text)
#                     st.success(f"Translation complete for {uploaded_file.name}")
#                 else:
#                     st.info("Document is already in English, no translation needed.")
                
#                 st.info(f"Chunking document and generating embeddings...")
#                 chunks = chunk_document(text)
#                 embeddings = generate_embeddings(chunks)
                
#                 st.info(f"Indexing {len(chunks)} chunks to OpenSearch...")
#                 bulk_data = []
#                 for chunk, embedding in zip(chunks, embeddings):
#                     document = {"text": chunk, "embeddings": embedding}
#                     bulk_data.append(json.dumps({"index": {"_index": INDEX_NAME}}))
#                     bulk_data.append(json.dumps(document))
#                 bulk_url = f"{OPENSEARCH_ENDPOINT}/_bulk"
#                 headers = {"Content-Type": "application/json"}
#                 response = requests.post(bulk_url, auth=awsauth, headers=headers, data="\n".join(bulk_data) + "\n")
                
#                 st.success(f"Indexed {uploaded_file.name} - Status Code: {response.status_code}")

