from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
import re
import requests
import datetime
import json
import pickle
import hashlib
import tempfile
import logging
from pathlib import Path
from urllib.parse import urlparse, unquote, parse_qs
from PyPDF2 import PdfReader
from docx import Document
import pandas as pd
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import spacy
from typing import List, Dict, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")
EMBEDDINGS_MODEL = "sentence-transformers/all-mpnet-base-v2"
FAISS_INDEX_DIR = "./faiss_indices"
METADATA_FILE = "document_metadata.json"
SUPPORTED_FILE_TYPES = {
    '.pdf': 'PDF Document',
    '.docx': 'Word Document', 
    '.doc': 'Word Document (Legacy)',
    '.txt': 'Text File',
    '.csv': 'CSV File',
    '.xlsx': 'Excel File',
    '.xls': 'Excel File (Legacy)',
    '.html': 'HTML File',
    '.htm': 'HTML File',
    '.md': 'Markdown File',
    '.json': 'JSON File',
    '.xml': 'XML File'
}
os.makedirs(FAISS_INDEX_DIR, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(title="Document Upload API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class URLRequest(BaseModel):
    urls: List[str]

class DocumentResponse(BaseModel):
    success: bool
    message: str
    index_name: Optional[str] = None
    chunks: Optional[int] = None

class DocumentStatus(BaseModel):
    total_docs: int
    total_chunks: int
    file_types: Dict[str, int]
    documents: Dict[str, Dict]

class TestURLRequest(BaseModel):
    url: str

# Initialize embeddings globally
try:
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDINGS_MODEL,
        model_kwargs={"device": "cpu"}
    )
    logger.info("Embeddings model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load embeddings model: {e}")
    embeddings = None

class URLProcessor:
    """Enhanced URL processor with Google Drive support"""
    
    @staticmethod
    def convert_google_drive_url(url: str) -> str:
        """Convert Google Drive share URL to direct download URL"""
        # Extract file ID from various Google Drive URL formats
        file_id = None
        
        # Format 1: https://drive.google.com/file/d/FILE_ID/view?usp=sharing
        match = re.search(r'/file/d/([a-zA-Z0-9-_]+)', url)
        if match:
            file_id = match.group(1)
        
        # Format 2: https://drive.google.com/open?id=FILE_ID
        elif 'open?id=' in url:
            parsed_url = urlparse(url)
            query_params = parse_qs(parsed_url.query)
            file_id = query_params.get('id', [None])[0]
        
        # Format 3: https://docs.google.com/document/d/FILE_ID/edit
        elif 'docs.google.com' in url:
            match = re.search(r'/d/([a-zA-Z0-9-_]+)', url)
            if match:
                file_id = match.group(1)
        
        if file_id:
            # Return direct download URL
            return f"https://drive.google.com/uc?export=download&id={file_id}"
        
        return url
    
    @staticmethod
    def is_google_drive_url(url: str) -> bool:
        """Check if URL is a Google Drive URL"""
        return any(domain in url.lower() for domain in [
            'drive.google.com',
            'docs.google.com/document',
            'docs.google.com/spreadsheets',
            'docs.google.com/presentation'
        ])
    
    @staticmethod
    def get_filename_from_google_drive(url: str, response_headers: dict) -> str:
        """Extract filename from Google Drive response headers or URL"""
        # Try to get filename from Content-Disposition header
        content_disposition = response_headers.get('content-disposition', '')
        if content_disposition:
            filename_match = re.search(r'filename[^;=\n]*=(([\'"]).*?\2|[^;\n]*)', content_disposition)
            if filename_match:
                filename = filename_match.group(1).strip('\'"')
                if filename:
                    return filename
        
        # Extract file ID and create a generic filename
        file_id_match = re.search(r'/file/d/([a-zA-Z0-9-_]+)', url)
        if file_id_match:
            file_id = file_id_match.group(1)
            return f"google_drive_file_{file_id}.pdf"  # Default to PDF
        
        return f"google_drive_file_{hashlib.md5(url.encode()).hexdigest()[:8]}.pdf"

def fetch_url_content(url: str, max_retries: int = 3) -> tuple:
    """
    Fetch content from URL with Google Drive support
    Returns: (content_bytes, filename, content_type)
    """
    original_url = url
    
    # Convert Google Drive URLs to direct download URLs
    if URLProcessor.is_google_drive_url(url):
        url = URLProcessor.convert_google_drive_url(url)
        logger.info(f"Converted Google Drive URL: {original_url} -> {url}")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    for attempt in range(max_retries):
        try:
            session = requests.Session()
            response = session.get(url, headers=headers, timeout=30, allow_redirects=True)
            
            # Handle Google Drive virus scan warning
            if 'drive.google.com' in response.url and 'virus scan warning' in response.text.lower():
                # Look for the actual download link in the response
                confirm_match = re.search(r'confirm=([^&"]+)', response.text)
                if confirm_match:
                    confirm_token = confirm_match.group(1)
                    download_url = f"{url}&confirm={confirm_token}"
                    response = session.get(download_url, headers=headers, timeout=30)
            
            response.raise_for_status()
            
            # Determine filename
            if URLProcessor.is_google_drive_url(original_url):
                filename = URLProcessor.get_filename_from_google_drive(original_url, response.headers)
            else:
                # Original filename logic
                content_disposition = response.headers.get('content-disposition', '')
                if content_disposition:
                    filename_match = re.search(r'filename[^;=\n]*=(([\'"]).*?\2|[^;\n]*)', content_disposition)
                    if filename_match:
                        filename = filename_match.group(1).strip('\'"')
                    else:
                        parsed_url = urlparse(original_url)
                        filename = unquote(Path(parsed_url.path).name)
                else:
                    parsed_url = urlparse(original_url)
                    filename = unquote(Path(parsed_url.path).name)
                
                if not filename or '.' not in filename:
                    content_type = response.headers.get('content-type', '').lower()
                    if 'html' in content_type:
                        filename = f"webpage_{hashlib.md5(original_url.encode()).hexdigest()[:8]}.html"
                    elif 'pdf' in content_type:
                        filename = f"document_{hashlib.md5(original_url.encode()).hexdigest()[:8]}.pdf"
                    else:
                        filename = f"content_{hashlib.md5(original_url.encode()).hexdigest()[:8]}.txt"
            
            content_type = response.headers.get('content-type', '')
            
            return response.content, filename, content_type
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Attempt {attempt + 1} failed for {url}: {str(e)}")
            if attempt == max_retries - 1:
                raise Exception(f"Failed to fetch URL after {max_retries} attempts: {str(e)}")
    
    raise Exception("Unexpected error in fetch_url_content")

class DocumentProcessor:
    """Handles processing of different document types"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )

    def extract_text_from_pdf(self, file_content: bytes) -> List[Dict]:
        """Extract text from PDF content"""
        try:
            from io import BytesIO
            pdf_file = BytesIO(file_content)
            pdf_reader = PdfReader(pdf_file)
            text_data = []
            
            for page_num, page in enumerate(pdf_reader.pages, 1):
                text = page.extract_text()
                if text and text.strip():
                    text_data.append({
                        'text': text.strip(),
                        'page': page_num,
                        'source_type': 'pdf_page'
                    })
            
            return text_data
        except Exception as e:
            logger.error(f"Error extracting PDF text: {str(e)}")
            raise Exception(f"Failed to extract PDF text: {str(e)}")

    def extract_text_from_docx(self, file_content: bytes) -> List[Dict]:
        """Extract text from DOCX content"""
        try:
            from io import BytesIO
            doc_file = BytesIO(file_content)
            doc = Document(doc_file)
            
            text_data = []
            full_text = []
            
            for para in doc.paragraphs:
                if para.text.strip():
                    full_text.append(para.text.strip())
            
            if full_text:
                text_data.append({
                    'text': '\n'.join(full_text),
                    'page': 1,
                    'source_type': 'docx_content'
                })
            
            return text_data
        except Exception as e:
            logger.error(f"Error extracting DOCX text: {str(e)}")
            raise Exception(f"Failed to extract DOCX text: {str(e)}")

    def extract_text_from_txt(self, file_content: bytes) -> List[Dict]:
        """Extract text from TXT content"""
        try:
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            text = None
            
            for encoding in encodings:
                try:
                    text = file_content.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if text is None:
                raise Exception("Could not decode text file with any supported encoding")
            
            return [{
                'text': text.strip(),
                'page': 1,
                'source_type': 'text_content'
            }]
        except Exception as e:
            logger.error(f"Error extracting TXT text: {str(e)}")
            raise Exception(f"Failed to extract TXT text: {str(e)}")

    def extract_text_from_csv(self, file_content: bytes) -> List[Dict]:
        """Extract text from CSV content"""
        try:
            from io import BytesIO
            csv_file = BytesIO(file_content)
            df = pd.read_csv(csv_file)
            
            text_parts = []
            text_parts.append(f"CSV Data with {len(df)} rows and {len(df.columns)} columns")
            text_parts.append(f"Columns: {', '.join(df.columns.tolist())}")
            text_parts.append("\nData Summary:")
            text_parts.append(df.to_string(max_rows=100))
            
            return [{
                'text': '\n'.join(text_parts),
                'page': 1,
                'source_type': 'csv_content'
            }]
        except Exception as e:
            logger.error(f"Error extracting CSV text: {str(e)}")
            raise Exception(f"Failed to extract CSV text: {str(e)}")

    def extract_text_from_excel(self, file_content: bytes) -> List[Dict]:
        """Extract text from Excel content"""
        try:
            from io import BytesIO
            excel_file = BytesIO(file_content)
            excel_data = pd.read_excel(excel_file, sheet_name=None)
            text_data = []
            
            for sheet_name, df in excel_data.items():
                text_parts = []
                text_parts.append(f"Sheet: {sheet_name}")
                text_parts.append(f"Data with {len(df)} rows and {len(df.columns)} columns")
                text_parts.append(f"Columns: {', '.join(df.columns.tolist())}")
                text_parts.append(df.to_string(max_rows=100))
                
                text_data.append({
                    'text': '\n'.join(text_parts),
                    'page': sheet_name,
                    'source_type': 'excel_sheet'
                })
            
            return text_data
        except Exception as e:
            logger.error(f"Error extracting Excel text: {str(e)}")
            raise Exception(f"Failed to extract Excel text: {str(e)}")

    def extract_text_from_html(self, file_content: bytes) -> List[Dict]:
        """Extract text from HTML content"""
        try:
            html_text = file_content.decode('utf-8')
            soup = BeautifulSoup(html_text, 'html.parser')
            
            for script in soup(["script", "style"]):
                script.decompose()
            
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return [{
                'text': text,
                'page': 1,
                'source_type': 'html_content'
            }]
        except Exception as e:
            logger.error(f"Error extracting HTML text: {str(e)}")
            raise Exception(f"Failed to extract HTML text: {str(e)}")

    def extract_text_from_json(self, file_content: bytes) -> List[Dict]:
        """Extract text from JSON content"""
        try:
            json_text = file_content.decode('utf-8')
            json_data = json.loads(json_text)
            
            formatted_json = json.dumps(json_data, indent=2, ensure_ascii=False)
            summary = f"JSON document with {len(str(json_data))} characters"
            if isinstance(json_data, dict):
                summary += f" and {len(json_data)} top-level keys: {', '.join(list(json_data.keys())[:10])}"
            elif isinstance(json_data, list):
                summary += f" containing {len(json_data)} items"
            
            full_text = f"{summary}\n\n{formatted_json}"
            
            return [{
                'text': full_text,
                'page': 1,
                'source_type': 'json_content'
            }]
        except Exception as e:
            logger.error(f"Error extracting JSON text: {str(e)}")
            raise Exception(f"Failed to extract JSON text: {str(e)}")

    def process_document(self, file_content: bytes, file_name: str, file_type: str) -> List[Dict]:
        """Process document based on file type"""
        file_ext = Path(file_name).suffix.lower()
        
        if file_ext == '.pdf':
            return self.extract_text_from_pdf(file_content)
        elif file_ext in ['.docx']:
            return self.extract_text_from_docx(file_content)
        elif file_ext == '.txt' or file_ext == '.md':
            return self.extract_text_from_txt(file_content)
        elif file_ext == '.csv':
            return self.extract_text_from_csv(file_content)
        elif file_ext in ['.xlsx', '.xls']:
            return self.extract_text_from_excel(file_content)
        elif file_ext in ['.html', '.htm']:
            return self.extract_text_from_html(file_content)
        elif file_ext == '.json':
            return self.extract_text_from_json(file_content)
        else:
            return self.extract_text_from_txt(file_content)

def sanitize_collection_name(name: str) -> str:
    """Sanitize document name for FAISS index directory"""
    name = Path(name).stem
    name = name.lower().replace(' ', '_')
    name = re.sub(r'[^a-z0-9_-]', '', name)
    name = re.sub(r'[_-]+', '_', name)
    
    if not name or not name[0].isalpha():
        name = 'doc_' + name
    if not name or not name[-1].isalnum():
        name = name + '0'
    if len(name) > 63:
        name = name[:62] + '0'
    if len(name) < 3:
        name = name + '_doc'
    
    return name
"""
def split_text_into_chunks(text_data: List[Dict]) -> tuple:
    #Split extracted text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = []
    metadatas = []
    
    for item in text_data:
        text_chunks = text_splitter.split_text(item['text'])
        for i, chunk in enumerate(text_chunks):
            chunks.append(chunk)
            metadatas.append({
                'page': item['page'],
                'chunk_size': len(chunk),
                'chunk_index': i,
                'source_type': item['source_type']
            })
    
    return chunks, metadatas
"""


def split_text_into_chunks(text_data: List[Dict], max_chunk_size=1000, chunk_overlap=200) -> Tuple[List[str], List[Dict]]:
    """Hybrid sentence-based + recursive splitter using spaCy."""
    
    chunks = []
    metadatas = []

    for item in text_data:
        doc = nlp(item['text'])
        sentences = [sent.text.strip() for sent in doc.sents]
        
        current_chunk = ""
        chunk_index = 0

        for sentence in sentences:
            if len(current_chunk) + len(sentence) > max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                    metadatas.append({
                        'page': item['page'],
                        'chunk_size': len(current_chunk),
                        'chunk_index': chunk_index,
                        'source_type': item['source_type']
                    })
                    chunk_index += 1
                current_chunk = sentence
            else:
                current_chunk += " " + sentence

        if current_chunk:
            chunks.append(current_chunk)
            metadatas.append({
                'page': item['page'],
                'chunk_size': len(current_chunk),
                'chunk_index': chunk_index,
                'source_type': item['source_type']
            })

    # Apply recursive splitter to any over-length chunk
    rec_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    final_chunks = []
    final_metadatas = []

    for i, chunk in enumerate(chunks):
        if len(chunk) > max_chunk_size:
            sub_chunks = rec_splitter.split_text(chunk)
            for j, sub_chunk in enumerate(sub_chunks):
                final_chunks.append(sub_chunk)
                final_metadatas.append({
                    'page': metadatas[i]['page'],
                    'chunk_size': len(sub_chunk),
                    'chunk_index': f"{metadatas[i]['chunk_index']}_{j}",
                    'source_type': metadatas[i]['source_type']
                })
        else:
            final_chunks.append(chunk)
            final_metadatas.append(metadatas[i])

    return final_chunks, final_metadatas





def create_faiss_index(chunks: List[str], metadatas: List[Dict], doc_name: str) -> Optional[str]:
    """Create FAISS index for the document chunks"""
    if not chunks or not embeddings:
        return None
    
    index_name = sanitize_collection_name(doc_name)
    
    counter = 1
    original_index_name = index_name
    while os.path.exists(os.path.join(FAISS_INDEX_DIR, index_name)):
        index_name = f"{original_index_name}_{counter}"
        counter += 1
    
    index_dir = os.path.join(FAISS_INDEX_DIR, index_name)
    
    try:
        # Create FAISS vectorstore with chunk-based embeddings
        vectorstore = FAISS.from_texts(
            texts=chunks,
            embedding=embeddings,
            metadatas=metadatas
        )
        
        os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
        os.makedirs(index_dir, exist_ok=True)
        vectorstore.save_local(index_dir)
        
        logger.info(f"Successfully created FAISS index: {index_name} with {len(chunks)} chunks")
        return index_name
    except Exception as e:
        logger.error(f"Error creating FAISS index: {str(e)}")
        return None
    




import requests
from typing import List

def split_text(text: str, max_chars: int = 3500) -> List[str]:
    """Split a long text into smaller parts for summarization."""
    parts = []
    while len(text) > max_chars:
        split_point = text.rfind('\n', 0, max_chars)  # Try splitting at newline
        if split_point == -1:
            split_point = max_chars
        parts.append(text[:split_point].strip())
        text = text[split_point:].strip()
    if text:
        parts.append(text)
    return parts

def summarize_chunk(chunk: str, model="llama3.2:latest", max_tokens=512) -> str:
    """Summarize one chunk using the local LLaMA model."""
    prompt = f"""
    Summarize the following policy document content in a concise and informative way, preserving key rules and procedures:

    {chunk}

    Summary:
    """
    try:
        response = requests.post("http://localhost:11434/api/generate", json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.3
            }
        }, timeout=60)

        if response.ok:
            return response.json().get("response", "").strip()
        else:
            return "Error: Could not summarize chunk"
    except Exception as e:
        return f"Error during summarization: {str(e)}"

def summarize_document_map_reduce(chunks: List[str], model="llama3.2:latest", max_tokens=512) -> str:
    """Full summarization using map-reduce on large document text chunks."""
    full_text = "\n\n".join(chunks)
    parts = split_text(full_text, max_chars=3500)

    # Step 1: Summarize each part
    partial_summaries = [summarize_chunk(part, model, max_tokens) for part in parts]

    # Step 2: Combine and summarize again
    combined_summary_text = "\n\n".join(partial_summaries)
    final_summary = summarize_chunk(combined_summary_text, model, max_tokens)

    return final_summary




def create_document_embeddings(index_name: str, doc_name: str, chunks: List[str], doc_summary: str = ""):
    """Create document-level embeddings for quick retrieval"""
    metadata_path = os.path.join(FAISS_INDEX_DIR, "document_embeddings")
    os.makedirs(metadata_path, exist_ok=True)

    try:
        # Generate summary if not provided
        if  len(doc_summary)==0:
            doc_summary = summarize_document_map_reduce(chunks, model="llama3.2:latest")
            
        # Create embedding text (document name + summary)
        embedding_text = f"{doc_name} {doc_summary}".strip()
        doc_embedding = embeddings.embed_documents([embedding_text])[0]
        
        # Save embedding
        doc_data = {
            "index_name": index_name,
            "original_name": doc_name,
            "summary": doc_summary,
            "embedding": doc_embedding,
            "created_at": str(datetime.datetime.now()),
            "num_chunks": len(chunks)
        }
        
        # Save to pickle file
        pickle_path = os.path.join(metadata_path, f"{index_name}.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(doc_data, f)
        
        logger.info(f"Created document embeddings for: {doc_name}")
        return doc_embedding
    except Exception as e:
        logger.error(f"Error creating document embeddings: {str(e)}")
        return None

def save_document_metadata(doc_name: str, index_name: str, file_type: str, num_pages: int, chunks: List[str], num_chunks: int, source_type: str = "upload", source_url: str = ""):
    """Save document metadata to a JSON file"""
    try:
        os.makedirs(os.path.dirname(METADATA_FILE) if os.path.dirname(METADATA_FILE) else ".", exist_ok=True)
        
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        # Ensure file_type is always a string
        if isinstance(file_type, list):
            file_type = file_type[0] if file_type else "Unknown"
        file_type = str(file_type)
        
        metadata[index_name] = {
            "original_name": doc_name,
            "file_type": file_type,
            "pages": num_pages,
            "chunks": num_chunks,
            "source_type": source_type,
            "source_url": source_url,
            "created_at": str(datetime.datetime.now()),
            "file_size": "N/A"
        }
        
        with open(METADATA_FILE, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create document-level embeddings for search
        #doc_summary = f"{file_type} document with {num_pages} pages/sections and {num_chunks} chunks"
        doc_summary=""
        create_document_embeddings(index_name, doc_name, chunks, doc_summary)
        
        logger.info(f"Saved metadata for document: {doc_name}")
    except Exception as e:
        logger.error(f"Error saving document metadata: {str(e)}")

def get_available_documents() -> Dict:
    """Get list of available documents"""
    if not os.path.exists(METADATA_FILE):
        return {}
    
    try:
        with open(METADATA_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error reading metadata file: {str(e)}")
        return {}

# Create necessary directories
os.makedirs(FAISS_INDEX_DIR, exist_ok=True)

# Initialize document processor
processor = DocumentProcessor()

@app.get("/")
async def root():
    return {"message": "Document Upload API is running"}

@app.get("/supported-types")
async def get_supported_types():
    return {"supported_types": SUPPORTED_FILE_TYPES}

@app.post("/upload-files", response_model=List[DocumentResponse])
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload and process multiple files"""
    if not embeddings:
        raise HTTPException(status_code=500, detail="Embeddings model not available")
    
    results = []
    
    for file in files:
        try:
            # Read file content
            file_content = await file.read()
            file_name = file.filename
            file_type = Path(file_name).suffix.lower()
            
            # Process document
            text_data = processor.process_document(file_content, file_name, file_type)
            
            if not text_data:
                results.append(DocumentResponse(
                    success=False,
                    message=f"No content extracted from {file_name}"
                ))
                continue
            
            # Split into chunks
            chunks, metadatas = split_text_into_chunks(text_data)
            
            if not chunks:
                results.append(DocumentResponse(
                    success=False,
                    message=f"No text chunks created for {file_name}"
                ))
                continue
            
            # Create FAISS index with chunk-based embeddings
            index_name = create_faiss_index(chunks, metadatas, file_name)
            
            if index_name:
                # Ensure file_type is a string from SUPPORTED_FILE_TYPES
                file_type_name = SUPPORTED_FILE_TYPES.get(file_type, "Unknown")
                
                # Save metadata
                save_document_metadata(
                    doc_name=file_name,
                    index_name=index_name,
                    file_type=file_type_name,  # This is now guaranteed to be a string
                    num_pages=len(text_data),
                    chunks=chunks,
                    num_chunks=len(chunks),
                    source_type="upload"
                )
                
                results.append(DocumentResponse(
                    success=True,
                    message=f"Successfully processed {file_name}",
                    index_name=index_name,
                    chunks=len(chunks)
                ))
            else:
                results.append(DocumentResponse(
                    success=False,
                    message=f"Failed to create index for {file_name}"
                ))
        
        except Exception as e:
            logger.error(f"Error processing {file_name}: {str(e)}")
            results.append(DocumentResponse(
                success=False,
                message=f"Error processing {file_name}: {str(e)}"
            ))
    
    return results

@app.post("/upload-urls", response_model=List[DocumentResponse])
async def upload_urls(request: URLRequest):
    """Upload and process URLs with Google Drive support"""
    if not embeddings:
        raise HTTPException(status_code=500, detail="Embeddings model not available")
    
    results = []
    
    for url in request.urls:
        try:
            logger.info(f"Processing URL: {url}")
            
            # Fetch URL content with enhanced Google Drive support
            content, filename, content_type = fetch_url_content(url)
            
            logger.info(f"Fetched content: {len(content)} bytes, filename: {filename}")
            
            # Process document
            text_data = processor.process_document(content, filename, content_type)
            
            if not text_data:
                results.append(DocumentResponse(
                    success=False,
                    message=f"No content extracted from {url}"
                ))
                continue
            
            # Split into chunks
            chunks, metadatas = split_text_into_chunks(text_data)
            
            if not chunks:
                results.append(DocumentResponse(
                    success=False,
                    message=f"No text chunks created for {url}"
                ))
                continue
            
            # Create FAISS index with chunk-based embeddings
            index_name = create_faiss_index(chunks, metadatas, filename)
            
            if index_name:
                # Determine file type - ensure it's always a string
                file_ext = Path(filename).suffix.lower()
                file_type = SUPPORTED_FILE_TYPES.get(file_ext, "Web Content")
                
                # Save metadata
                save_document_metadata(
                    doc_name=filename,
                    index_name=index_name,
                    file_type=file_type,  # This is now guaranteed to be a string
                    num_pages=len(text_data),
                    chunks=chunks,
                    num_chunks=len(chunks),
                    source_type="url",
                    source_url=url
                )
                
                results.append(DocumentResponse(
                    success=True,
                    message=f"Successfully processed {url}",
                    index_name=index_name,
                    chunks=len(chunks)
                ))
                
                logger.info(f"Successfully processed {url} -> {index_name} with {len(chunks)} chunks")
            else:
                results.append(DocumentResponse(
                    success=False,
                    message=f"Failed to create index for {url}"
                ))
        
        except Exception as e:
            logger.error(f"Error processing {url}: {str(e)}")
            results.append(DocumentResponse(
                success=False,
                message=f"Error processing {url}: {str(e)}"
            ))
    
    return results

@app.post("/test-url", response_model=DocumentResponse)
async def test_url(request: TestURLRequest):
    """Test URL accessibility and content type"""
    try:
        url = request.url
        logger.info(f"Testing URL: {url}")
        
        # Check if it's a Google Drive URL and convert if needed
        if URLProcessor.is_google_drive_url(url):
            converted_url = URLProcessor.convert_google_drive_url(url)
            logger.info(f"Converted Google Drive URL: {url} -> {converted_url}")
            url = converted_url
        
        # Test URL accessibility
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.head(url, headers=headers, timeout=10, allow_redirects=True)
        
        if response.status_code == 200:
            content_type = response.headers.get('content-type', 'Unknown')
            content_length = response.headers.get('content-length', 'Unknown')
            
            return DocumentResponse(
                success=True,
                message=f"URL is accessible. Content-Type: {content_type}, Content-Length: {content_length}"
            )
        else:
            return DocumentResponse(
                success=False,
                message=f"URL returned status code: {response.status_code}"
            )
    
    except Exception as e:
        logger.error(f"Error testing URL {request.url}: {str(e)}")
        return DocumentResponse(
            success=False,
            message=f"Error testing URL: {str(e)}"
        )

@app.get("/documents", response_model=DocumentStatus)
async def get_documents():
    """Get status of all uploaded documents"""
    try:
        documents = get_available_documents()
        
        if not documents:
            return DocumentStatus(
                total_docs=0,
                total_chunks=0,
                file_types={},
                documents={}
            )
        
        total_docs = len(documents)
        total_chunks = sum(doc.get('chunks', 0) for doc in documents.values())
        
        # Count file types
        file_types = {}
        for doc in documents.values():
            file_type = doc.get('file_type', 'Unknown')
            file_types[file_type] = file_types.get(file_type, 0) + 1
        
        return DocumentStatus(
            total_docs=total_docs,
            total_chunks=total_chunks,
            file_types=file_types,
            documents=documents
        )
    
    except Exception as e:
        logger.error(f"Error getting document status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving documents: {str(e)}")

@app.delete("/documents/{index_name}")
async def delete_document(index_name: str):
    """Delete a specific document and its index"""
    try:
        # Remove FAISS index directory
        index_dir = os.path.join(FAISS_INDEX_DIR, index_name)
        if os.path.exists(index_dir):
            import shutil
            shutil.rmtree(index_dir)
            logger.info(f"Deleted FAISS index directory: {index_dir}")
        
        # Remove document embedding
        embedding_file = os.path.join(FAISS_INDEX_DIR, "document_embeddings", f"{index_name}.pkl")
        if os.path.exists(embedding_file):
            os.remove(embedding_file)
            logger.info(f"Deleted document embedding: {embedding_file}")
        
        # Remove from metadata
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, 'r') as f:
                metadata = json.load(f)
            
            if index_name in metadata:
                doc_name = metadata[index_name].get('original_name', index_name)
                del metadata[index_name]
                
                with open(METADATA_FILE, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                return {"success": True, "message": f"Successfully deleted document: {doc_name}"}
            else:
                return {"success": False, "message": f"Document {index_name} not found in metadata"}
        else:
            return {"success": False, "message": "No metadata file found"}
    
    except Exception as e:
        logger.error(f"Error deleting document {index_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

@app.delete("/documents")
async def delete_all_documents():
    """Delete all documents and indices"""
    try:
        # Remove all FAISS indices
        if os.path.exists(FAISS_INDEX_DIR):
            import shutil
            shutil.rmtree(FAISS_INDEX_DIR)
            os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
            logger.info("Deleted all FAISS indices")
        
        # Remove metadata file
        if os.path.exists(METADATA_FILE):
            os.remove(METADATA_FILE)
            logger.info("Deleted metadata file")
        
        return {"success": True, "message": "All documents deleted successfully"}
    
    except Exception as e:
        logger.error(f"Error deleting all documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting all documents: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "embeddings_loaded": embeddings is not None,
        "faiss_dir_exists": os.path.exists(FAISS_INDEX_DIR),
        "timestamp": str(datetime.datetime.now())
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")