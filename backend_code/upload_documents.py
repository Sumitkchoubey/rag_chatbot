import streamlit as st
import os
import re
import requests
import datetime
import numpy as np
import mimetypes
import hashlib
from pathlib import Path
from urllib.parse import urlparse, unquote
from PyPDF2 import PdfReader
from docx import Document
import pandas as pd
from bs4 import BeautifulSoup
import tempfile
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import json
import pickle
from typing import List, Dict, Tuple, Optional
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
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
            # Try different encodings
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
            
            # Convert DataFrame to readable text
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
            
            # Read all sheets
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
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text
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
            
            # Convert JSON to readable text
            formatted_json = json.dumps(json_data, indent=2, ensure_ascii=False)
            
            # Also create a summary
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
            # Fallback to text extraction
            return self.extract_text_from_txt(file_content)


class URLProcessor:
    """Handles processing of URLs"""
    
    @staticmethod
    def fetch_url_content(url: str) -> Tuple[bytes, str, str]:
        """Fetch content from URL"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Determine content type
            content_type = response.headers.get('content-type', '').lower()
            
            # Extract filename from URL
            parsed_url = urlparse(url)
            filename = unquote(Path(parsed_url.path).name)
            if not filename or '.' not in filename:
                if 'html' in content_type:
                    filename = f"webpage_{hashlib.md5(url.encode()).hexdigest()[:8]}.html"
                else:
                    filename = f"content_{hashlib.md5(url.encode()).hexdigest()[:8]}.txt"
            
            return response.content, filename, content_type
        except Exception as e:
            logger.error(f"Error fetching URL {url}: {str(e)}")
            raise Exception(f"Failed to fetch URL: {str(e)}")


def sanitize_collection_name(name: str) -> str:
    """Sanitize document name for FAISS index directory"""
    # Remove file extension
    name = Path(name).stem
    
    # Convert to lowercase and replace spaces
    name = name.lower().replace(' ', '_')
    
    # Remove special characters except underscore and hyphen
    name = re.sub(r'[^a-z0-9_-]', '', name)
    
    # Replace multiple underscores/hyphens with single underscore
    name = re.sub(r'[_-]+', '_', name)
    
    # Ensure it starts with a letter
    if not name or not name[0].isalpha():
        name = 'doc_' + name
    
    # Ensure it ends with alphanumeric
    if not name or not name[-1].isalnum():
        name = name + '0'
    
    # Limit length
    if len(name) > 63:
        name = name[:62] + '0'
    
    # Ensure minimum length
    if len(name) < 3:
        name = name + '_doc'
    
    return name


def split_text_into_chunks(text_data: List[Dict]) -> Tuple[List[str], List[Dict]]:
    """Split extracted text into manageable chunks"""
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


def create_faiss_index(chunks: List[str], metadatas: List[Dict], doc_name: str, embeddings) -> Optional[str]:
    """Create FAISS index for the document"""
    if not chunks:
        st.error("❌ No text chunks to store. Skipping FAISS index creation.")
        return None
    
    index_name = sanitize_collection_name(doc_name)
    
    # Ensure unique index name
    counter = 1
    original_index_name = index_name
    while os.path.exists(os.path.join(FAISS_INDEX_DIR, index_name)):
        index_name = f"{original_index_name}_{counter}"
        counter += 1
    
    index_dir = os.path.join(FAISS_INDEX_DIR, index_name)
    
    try:
        vectorstore = FAISS.from_texts(
            texts=chunks,
            embedding=embeddings,
            metadatas=metadatas
        )
        
        # Create directory if it doesn't exist
        os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
        os.makedirs(index_dir, exist_ok=True)
        
        # Save the FAISS index
        vectorstore.save_local(index_dir)
        
        logger.info(f"Successfully created FAISS index: {index_name}")
        return index_name
    except Exception as e:
        logger.error(f"Error creating FAISS index: {str(e)}")
        st.error(f"❌ Error creating FAISS index: {str(e)}")
        return None


def create_document_embeddings(index_name: str, doc_name: str, doc_summary: str = ""):
    """Create document-level embeddings for quick retrieval"""
    metadata_path = os.path.join(FAISS_INDEX_DIR, "document_embeddings")
    os.makedirs(metadata_path, exist_ok=True)

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDINGS_MODEL,
            model_kwargs={"device": "cpu"}
        )
        
        # Create embedding text (document name + summary)
        embedding_text = f"{doc_name} {doc_summary}".strip()
        doc_embedding = embeddings.embed_documents([embedding_text])[0]
        
        # Save embedding
        doc_data = {
            "index_name": index_name,
            "original_name": doc_name,
            "summary": doc_summary,
            "embedding": doc_embedding,
            "created_at": str(datetime.datetime.now())
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


def save_document_metadata(doc_name: str, index_name: str, file_type: str, num_pages: int, num_chunks: int, source_type: str = "upload", source_url: str = ""):
    """Save document metadata to a JSON file"""
    try:
        os.makedirs(os.path.dirname(METADATA_FILE) if os.path.dirname(METADATA_FILE) else ".", exist_ok=True)
        
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
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
        doc_summary = f"{file_type} document with {num_pages} pages/sections and {num_chunks} chunks"
        create_document_embeddings(index_name, doc_name, doc_summary)
        
        logger.info(f"Saved metadata for document: {doc_name}")
    except Exception as e:
        logger.error(f"Error saving document metadata: {str(e)}")
        st.error(f"❌ Error saving document metadata: {str(e)}")


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


def upload_page():
    """Enhanced Document Upload Page"""
    st.title("📚 Universal Document Upload System")
    st.markdown("Upload various file types or provide URLs to process documents into searchable indices.")
    
    # Initialize embeddings
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDINGS_MODEL,
            model_kwargs={"device": "cpu"}
        )
    except Exception as e:
        st.error(f"❌ Error initializing embeddings: {str(e)}")
        return
    
    # Create tabs for different upload methods
    tab1, tab2 = st.tabs(["📁 File Upload", "🌐 URL Upload"])
    
    with tab1:
        st.markdown("### Upload Files")
        st.markdown(f"**Supported file types:** {', '.join(SUPPORTED_FILE_TYPES.values())}")
        
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            type=list(SUPPORTED_FILE_TYPES.keys())[0].replace('.', ''),
            accept_multiple_files=True,
            help="Select one or more files to process and index"
        )
        
        if uploaded_files:
            process_uploaded_files(uploaded_files, embeddings)
    
    with tab2:
        st.markdown("### Upload from URLs")
        st.markdown("Enter URLs to web pages, documents, or files to download and process.")
        
        # URL input methods
        url_input_method = st.radio(
            "Choose input method:",
            ["Single URL", "Multiple URLs (one per line)"]
        )
        
        urls = []
        if url_input_method == "Single URL":
            url = st.text_input("Enter URL:", placeholder="https://example.com/document.pdf")
            if url:
                urls = [url]
        else:
            url_text = st.text_area(
                "Enter URLs (one per line):",
                placeholder="https://example.com/doc1.pdf\nhttps://example.com/doc2.html\nhttps://example.com/page3"
            )
            if url_text:
                urls = [url.strip() for url in url_text.split('\n') if url.strip()]
        
        if urls and st.button("Process URLs"):
            process_urls(urls, embeddings)
    
    # Display current documents
    display_document_status()


def process_uploaded_files(uploaded_files, embeddings):
    """Process uploaded files"""
    st.markdown("### Processing Uploaded Files")
    
    progress_bar = st.progress(0)
    progress_text = st.empty()
    status_container = st.container()
    
    processor = DocumentProcessor()
    successful_uploads = 0
    
    for i, uploaded_file in enumerate(uploaded_files):
        file_name = uploaded_file.name
        file_type = Path(file_name).suffix.lower()
        
        progress_text.text(f"Processing {file_name}...")
        
        try:
            # Read file content
            file_content = uploaded_file.read()
            
            # Process document
            text_data = processor.process_document(file_content, file_name, file_type)
            
            if not text_data:
                with status_container:
                    st.warning(f"⚠️ No content extracted from {file_name}")
                continue
            
            # Split into chunks
            chunks, metadatas = split_text_into_chunks(text_data)
            
            if not chunks:
                with status_container:
                    st.warning(f"⚠️ No text chunks created for {file_name}")
                continue
            
            # Create FAISS index
            index_name = create_faiss_index(chunks, metadatas, file_name, embeddings)
            
            if index_name:
                # Save metadata
                save_document_metadata(
                    file_name,
                    index_name,
                    SUPPORTED_FILE_TYPES.get(file_type, "Unknown"),
                    len(text_data),
                    len(chunks),
                    "upload"
                )
                
                with status_container:
                    st.success(f"✅ Successfully processed {file_name} → Index: {index_name}")
                successful_uploads += 1
            else:
                with status_container:
                    st.error(f"❌ Failed to create index for {file_name}")
        
        except Exception as e:
            with status_container:
                st.error(f"❌ Error processing {file_name}: {str(e)}")
        
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    progress_text.text(f"Completed! Successfully processed {successful_uploads}/{len(uploaded_files)} files.")


def process_urls(urls, embeddings):
    """Process URLs"""
    st.markdown("### Processing URLs")
    
    progress_bar = st.progress(0)
    progress_text = st.empty()
    status_container = st.container()
    
    processor = DocumentProcessor()
    url_processor = URLProcessor()
    successful_uploads = 0
    
    for i, url in enumerate(urls):
        progress_text.text(f"Processing {url}...")
        
        try:
            # Fetch URL content
            file_content, file_name, content_type = url_processor.fetch_url_content(url)
            
            # Process document
            text_data = processor.process_document(file_content, file_name, content_type)
            
            if not text_data:
                with status_container:
                    st.warning(f"⚠️ No content extracted from {url}")
                continue
            
            # Split into chunks
            chunks, metadatas = split_text_into_chunks(text_data)
            
            if not chunks:
                with status_container:
                    st.warning(f"⚠️ No text chunks created for {url}")
                continue
            
            # Create FAISS index
            index_name = create_faiss_index(chunks, metadatas, file_name, embeddings)
            
            if index_name:
                # Determine file type
                file_ext = Path(file_name).suffix.lower()
                file_type = SUPPORTED_FILE_TYPES.get(file_ext, "Web Content")
                
                # Save metadata
                save_document_metadata(
                    file_name,
                    index_name,
                    file_type,
                    len(text_data),
                    len(chunks),
                    "url",
                    url
                )
                
                with status_container:
                    st.success(f"✅ Successfully processed {url} → Index: {index_name}")
                successful_uploads += 1
            else:
                with status_container:
                    st.error(f"❌ Failed to create index for {url}")
        
        except Exception as e:
            with status_container:
                st.error(f"❌ Error processing {url}: {str(e)}")
        
        progress_bar.progress((i + 1) / len(urls))
    
    progress_text.text(f"Completed! Successfully processed {successful_uploads}/{len(urls)} URLs.")


def display_document_status():
    """Display current document status"""
    st.markdown("### 📊 Document Library Status")
    
    documents = get_available_documents()
    
    if not documents:
        st.info("No documents uploaded yet. Upload some documents to get started!")
        return
    
    # Create a summary
    total_docs = len(documents)
    total_chunks = sum(doc.get('chunks', 0) for doc in documents.values())
    file_types = {}
    
    for doc in documents.values():
        file_type = doc.get('file_type', 'Unknown')
        file_types[file_type] = file_types.get(file_type, 0) + 1
    
    # Display summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Documents", total_docs)
    with col2:
        st.metric("Total Chunks", total_chunks)
    with col3:
        st.metric("File Types", len(file_types))
    
    # Display file type breakdown
    if file_types:
        st.markdown("**File Type Distribution:**")
        for file_type, count in file_types.items():
            st.write(f"• {file_type}: {count} document(s)")
    
    # Display document list
    with st.expander("View All Documents", expanded=False):
        for index_name, doc_info in documents.items():
            st.markdown(f"**{doc_info['original_name']}**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.write(f"Type: {doc_info.get('file_type', 'N/A')}")
            with col2:
                st.write(f"Pages: {doc_info.get('pages', 'N/A')}")
            with col3:
                st.write(f"Chunks: {doc_info.get('chunks', 'N/A')}")
            with col4:
                st.write(f"Source: {doc_info.get('source_type', 'N/A')}")
            
            if doc_info.get('source_url'):
                st.write(f"URL: {doc_info['source_url']}")
            
            st.write(f"Created: {doc_info.get('created_at', 'N/A')}")
            st.write(f"Index: `{index_name}`")
            st.markdown("---")


def main():
    """Main application function"""
    st.set_page_config(
        page_title="Universal Document Upload System",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Sidebar information
    with st.sidebar:
        st.title("📚 Document Upload System")
        st.markdown("### Features")
        st.markdown("""
        • **Multi-format support**: PDF, DOCX, TXT, CSV, Excel, HTML, JSON, XML, Markdown
        • **URL processing**: Download and process web content
        • **FAISS indexing**: Production-ready vector storage
        • **Metadata tracking**: Complete document management
        • **Chunk optimization**: Smart text splitting for better retrieval
        """)
        
        st.markdown("### System Status")
        # Check system requirements
        if os.path.exists(FAISS_INDEX_DIR):
            st.success("✅ FAISS directory ready")
        else:
            st.info("📁 FAISS directory will be created")
        
        # Show embeddings model
        st.info(f"🤖 Embeddings: {EMBEDDINGS_MODEL}")
        
        # Show supported formats
        with st.expander("Supported Formats"):
            for ext, desc in SUPPORTED_FILE_TYPES.items():
                st.write(f"• {ext}: {desc}")
    
    # Create necessary directories
    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
    
    # Main upload page
    upload_page()


if __name__ == "__main__":
    main()