import os
import numpy as np
import pickle
import re
from typing import List, Dict, Any, Optional
import requests
import json
import uvicorn
import uuid
from difflib import SequenceMatcher
from fuzzywuzzy import fuzz, process
import Levenshtein
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

app = FastAPI(title="DocumentMind", description="Chat with your documents using AI")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
EMBEDDINGS_MODEL = "sentence-transformers/all-mpnet-base-v2"
FAISS_INDEX_DIR = "./faiss_indices"
METADATA_FILE = "document_metadata.json"
OLLAMA_API_BASE = "http://localhost:11434"

# Document keyword mapping for direct queries
DOCUMENT_KEYWORDS = {
    'appraisal': 'appraisal_policy',
    'leave': 'leave_policy_2025', 
    'performance': 'performance_improvement_plan',
    'benefits': 'employees_benefits_plan174068328467c0b8149f1e0',
    'mediclaim': 'mediclaim',
    'posh': 'posh',
    'accommodation': 'accommodation_rr',
    'finance': 'general_finance_norms174067985467c0aaae3ecfb'
}

# Models
class Message(BaseModel):
    id: str = Field(..., description="Unique message ID")
    role: str = Field(..., description="Role can be either 'user' or 'assistant'")
    content: str = Field(..., description="The content of the message")

class ChatRequest(BaseModel):
    message: str = Field(..., description="User's message/question")
    model_name: str = Field(default="llama3.2:latest", description="Model to use for response generation")
    session_id: Optional[str] = Field(default=None, description="Browser session ID")
    use_chat_history: bool = Field(default=True, description="Whether to use chat history in context")

class ChatResponse(BaseModel):
    answer: str = Field(..., description="Assistant's response")
    page_numbers: List[int] = Field(default_factory=list, description="Source page numbers")
    document_name: Optional[str] = Field(default=None, description="Source document name")
    session_id: str = Field(..., description="Browser session ID for tracking the chat")
    success: bool = Field(default=True, description="Whether the request was successful")
    error: Optional[str] = Field(default=None, description="Error message if any")
    used_chat_history: bool = Field(default=True, description="Whether chat history was used")
    match_type: Optional[str] = Field(default=None, description="Type of document matching used")
    similarity_score: Optional[float] = Field(default=None, description="Document similarity score")

class ConversationData(BaseModel):
    session_id: str
    messages: List[Message] = Field(default_factory=list)
    document_context: Optional[str] = Field(default=None)

# In-memory storage for chat history based on browser session
# Key: session_id (browser-specific), Value: ConversationData
session_conversations = {}

def generate_session_id() -> str:
    """Generate a new unique session ID"""
    return str(uuid.uuid4())

def generate_message_id() -> str:
    """Generate a new unique message ID"""
    return str(uuid.uuid4())

def get_or_create_session(session_id: Optional[str] = None) -> ConversationData:
    """Get existing session or create new one"""
    if session_id and session_id in session_conversations:
        return session_conversations[session_id]
    
    # Create new session
    new_session_id = session_id if session_id else generate_session_id()
    conversation = ConversationData(session_id=new_session_id)
    session_conversations[new_session_id] = conversation
    return conversation

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    return dot_product / (norm_a * norm_b)

def calculate_fuzzy_similarity(query: str, target: str) -> float:
    """Calculate fuzzy similarity using multiple methods"""
    if not query or not target:
        return 0.0
    
    query = query.lower().strip()
    target = target.lower().strip()
    
    # If strings are identical, return perfect match
    if query == target:
        return 1.0
    
    # Method 1: Levenshtein distance (edit distance)
    max_len = max(len(query), len(target))
    if max_len == 0:
        return 1.0
    levenshtein_sim = 1 - (Levenshtein.distance(query, target) / max_len)
    
    # Method 2: Sequence matcher (difflib)
    sequence_sim = SequenceMatcher(None, query, target).ratio()
    
    # Method 3: Fuzzy ratio
    fuzzy_ratio = fuzz.ratio(query, target) / 100.0
    
    # Method 4: Partial ratio for substring matching
    partial_ratio = fuzz.partial_ratio(query, target) / 100.0
    
    # Method 5: Token sort ratio (handles word order differences)
    token_sort_ratio = fuzz.token_sort_ratio(query, target) / 100.0
    
    # Combine scores with weights (you can adjust these weights)
    combined_score = (
        levenshtein_sim * 0.25 +
        sequence_sim * 0.25 +
        fuzzy_ratio * 0.2 +
        partial_ratio * 0.15 +
        token_sort_ratio * 0.15
    )
    
    return combined_score

def extract_clean_document_name(filename: str) -> str:
    """Extract clean document name from filename for fuzzy matching"""
    # Remove file extension
    name = os.path.splitext(filename)[0]
    
    # Remove common suffixes and numbers
    name = re.sub(r'_\d+$', '', name)  # Remove trailing numbers
    name = re.sub(r'[_-]', ' ', name)  # Replace underscores and hyphens with spaces
    name = re.sub(r'\s+', ' ', name).strip().lower()  # Normalize spaces and lowercase
    
    return name

def extract_keywords_from_query(query: str) -> List[str]:
    """Extract meaningful keywords from query"""
    # Remove common stop words and extract meaningful terms
    stop_words = {'the', 'is', 'at', 'which', 'on', 'and', 'or', 'but', 'in', 'with', 'a', 'an', 'are', 'as', 'be', 'been', 'by', 'for', 'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'to', 'was', 'were', 'will', 'with', 'what', 'when', 'where', 'how', 'about', 'can', 'could', 'should', 'would', 'my', 'me', 'i', 'you', 'your', 'tell', 'show', 'give', 'get', 'find', 'help'}
    
    # Extract words that are at least 3 characters long
    words = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
    keywords = [word for word in words if word not in stop_words]
    
    return keywords

def check_ollama_status():
    """Check if Ollama server is running and get available models"""
    try:
        response = requests.get(f"{OLLAMA_API_BASE}/api/tags")
        if response.status_code == 200:
            available_models = [model["name"] for model in response.json()["models"]]
            return True, available_models
        else:
            return False, []
    except Exception:
        return False, []

def find_relevant_documents(query, top_k=5):
    """Enhanced document finding with fuzzy matching for misspelled queries"""
    metadata_path = os.path.join(FAISS_INDEX_DIR, "document_embeddings")
    if not os.path.exists(metadata_path):
        return None, "No document embeddings found."
    
    # Get all document embeddings
    docs_data = []
    for file in os.listdir(metadata_path):
        if file.endswith(".pkl"):
            file_path = os.path.join(metadata_path, file)
            with open(file_path, "rb") as f:
                doc_data = pickle.load(f)
                docs_data.append(doc_data)
    
    if not docs_data:
        return None, "No document embeddings found."
    
    query_lower = query.lower().strip()
    query_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', query_lower))
    
    print(f"Processing query: '{query}' with extracted words: {query_words}")
    
    # Step 1: Check for exact keyword matches first (highest priority)
    for keyword, doc_index in DOCUMENT_KEYWORDS.items():
        if keyword in query_lower:
            for doc in docs_data:
                if doc["index_name"] == doc_index:
                    print(f"✓ Exact keyword match: '{query}' -> {doc['original_name']} (keyword: {keyword})")
                    return [{
                        "index_name": doc["index_name"],
                        "original_name": doc["original_name"],
                        "similarity": 1.0,
                        "match_type": "exact_keyword",
                        "matched_term": keyword
                    }], None
    
    # Step 2: Fuzzy matching with keywords for misspelled queries
    fuzzy_threshold = 0.65  # Adjustable threshold for fuzzy matching
    best_fuzzy_match = None
    best_fuzzy_score = 0
    matched_keyword = None
    
    print(f"Checking fuzzy matches with keywords (threshold: {fuzzy_threshold})")
    for keyword, doc_index in DOCUMENT_KEYWORDS.items():
        fuzzy_score = calculate_fuzzy_similarity(query_lower, keyword)
        print(f"  '{query_lower}' vs '{keyword}': {fuzzy_score:.3f}")
        
        if fuzzy_score >= fuzzy_threshold and fuzzy_score > best_fuzzy_score:
            # Find the corresponding document
            for doc in docs_data:
                if doc["index_name"] == doc_index:
                    best_fuzzy_match = {
                        "index_name": doc["index_name"],
                        "original_name": doc["original_name"],
                        "similarity": fuzzy_score,
                        "match_type": "fuzzy_keyword",
                        "matched_term": keyword
                    }
                    best_fuzzy_score = fuzzy_score
                    matched_keyword = keyword
                    break
    
    if best_fuzzy_match:
        print(f"✓ Fuzzy keyword match: '{query}' -> {best_fuzzy_match['original_name']} "
              f"(matched: {matched_keyword}, score: {best_fuzzy_score:.3f})")
        return [best_fuzzy_match], None
    
    # Step 3: Fuzzy matching with document names
    print("Checking fuzzy matches with document names")
    name_matches = []
    for doc in docs_data:
        clean_name = extract_clean_document_name(doc["original_name"])
        name_score = calculate_fuzzy_similarity(query_lower, clean_name)
        print(f"  '{query_lower}' vs '{clean_name}': {name_score:.3f}")
        
        if name_score >= fuzzy_threshold:
            name_matches.append({
                "index_name": doc["index_name"],
                "original_name": doc["original_name"],
                "similarity": name_score,
                "match_type": "fuzzy_name",
                "matched_term": clean_name
            })
    
    if name_matches:
        # Sort by similarity and return the best match
        name_matches.sort(key=lambda x: x["similarity"], reverse=True)
        best_name_match = name_matches[0]
        print(f"✓ Fuzzy name match: '{query}' -> {best_name_match['original_name']} "
              f"(score: {best_name_match['similarity']:.3f})")
        return [best_name_match], None
    
    # Step 4: Check for partial keyword matches with lower threshold
    lower_threshold = 0.5
    print(f"Checking partial matches with lower threshold ({lower_threshold})")
    
    query_keywords = extract_keywords_from_query(query)
    for query_word in query_keywords:
        for keyword, doc_index in DOCUMENT_KEYWORDS.items():
            fuzzy_score = calculate_fuzzy_similarity(query_word, keyword)
            if fuzzy_score >= lower_threshold:
                for doc in docs_data:
                    if doc["index_name"] == doc_index:
                        print(f"✓ Partial keyword match: '{query_word}' -> '{keyword}' -> {doc['original_name']} "
                              f"(score: {fuzzy_score:.3f})")
                        return [{
                            "index_name": doc["index_name"],
                            "original_name": doc["original_name"],
                            "similarity": fuzzy_score,
                            "match_type": "partial_keyword",
                            "matched_term": f"{query_word} -> {keyword}"
                        }], None
    
    # Step 5: Exact filename word matching (original logic)
    print("Checking exact word matches in filenames")
    for doc in docs_data:
        filename_lower = doc["original_name"].lower()
        filename_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', filename_lower))
        
        matches = query_words.intersection(filename_words)
        if matches:
            match_strength = len(matches) / len(query_words) if query_words else 0
            
            if match_strength >= 0.5:
                print(f"✓ Word match found: '{query}' -> {doc['original_name']} "
                      f"(strength: {match_strength:.2f}, matches: {matches})")
                return [{
                    "index_name": doc["index_name"],
                    "original_name": doc["original_name"],
                    "similarity": match_strength,
                    "match_type": "word_match",
                    "matched_term": str(matches)
                }], None
    
    # Step 6: Fallback to semantic similarity (original method)
    print(f"No fuzzy matches found, using semantic similarity for '{query}'")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDINGS_MODEL,
        model_kwargs={"device": "cpu"}
    )
    query_embedding = embeddings.embed_query(query)
    
    docs_with_scores = []
    for doc in docs_data:
        similarity = cosine_similarity(query_embedding, doc["embedding"])
        docs_with_scores.append({
            "index_name": doc["index_name"],
            "original_name": doc["original_name"],
            "similarity": similarity,
            "match_type": "semantic_similarity",
            "matched_term": "embedding_based"
        })
    
    print(f"Semantic similarity results for '{query}':")
    for doc in docs_with_scores:
        print(f"  {doc['original_name']}: {doc['similarity']:.3f}")
    
    docs_with_scores.sort(key=lambda x: x["similarity"], reverse=True)
    return docs_with_scores[:top_k], None

def load_faiss_index(index_name, embeddings):
    """Load FAISS index for a document"""
    index_dir = os.path.join(FAISS_INDEX_DIR, index_name)
    
    if not os.path.exists(index_dir):
        return None, f"Index for {index_name} not found."
    
    try:
        vectorstore = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
        return vectorstore, None
    except Exception as e:
        return None, f"Error loading FAISS index: {str(e)}"

def build_chat_history_context(messages, max_history=5, use_history=True):
    """Build context from chat history for the prompt"""
    if not use_history:
        return ""
    
    # Only use the most recent messages to avoid context length issues
    recent_messages = messages[-max_history:] if len(messages) > max_history else messages
    
    # Exclude the current user message (last message) from history
    if recent_messages and recent_messages[-1].role == "user":
        recent_messages = recent_messages[:-1]
    
    history_text = ""
    for msg in recent_messages:
        role = "User" if msg.role == "user" else "Assistant"
        history_text += f"{role}: {msg.content}\n\n"
    
    return history_text

def answer_question(vectorstore, question, conversation_history="", model_name="llama3.2:latest", use_chat_history=False):
    """Answer a question using the RAG approach with Ollama, with optional chat history"""

    try:
        # Create retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

        # Retrieve relevant documents
        relevant_docs = retriever.get_relevant_documents(question)

        # Combine document texts into one context string
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        print(f"Retrieved context length: {len(context)} characters")
        
        # Choose prompt template based on whether to use chat history
        if use_chat_history and conversation_history.strip():
            custom_prompt = PromptTemplate.from_template("""
            You are DocumentMind, a helpful AI assistant that specializes in answering questions about documents.

            Always respond in a conversational, friendly manner as if you're chatting with the user.

            Previous conversation:
            {conversation_history}

            Context from documents:
            {context}

            User question:
            {question}

            Your answer should be comprehensive yet conversational. Use only the information from the context above — 
            do not use external knowledge or make assumptions beyond what's provided.

            Be especially careful to distinguish between:
            - How often a benefit is given (frequency), and
            - When or how the benefit value changes (amount or eligibility criteria).

            If the context mentions that a benefit increases after a certain time, do not assume that means the benefit is only given at those intervals. 
            Clarify if the timing of the benefit is mentioned separately.

            If the document does not specify something (such as timing, frequency, or exceptions), do not guess or assume. 
            Just say the document does not provide that information.

            Do not invent new details, scenarios, or policy variations unless they are explicitly stated in the document.

            Your response must be at least 3 to 5 sentences long.

            After your response, include a confidence score (1–10) based on how well the context supports your answer.

            Format:
            Answer: <your answer here>

            Confidence Score: <number>/10 – <brief explanation>
            """)
        else:
            # Prompt without chat history
            custom_prompt = PromptTemplate.from_template("""
            You are DocumentMind, a helpful AI assistant that specializes in answering questions about documents.

            Always respond in a conversational, friendly manner as if you're chatting with the user.

            Context from documents:
            {context}

            User question:
            {question}

            Your answer should be comprehensive yet conversational. Use only the information from the context above — 
            do not use external knowledge or make assumptions beyond what's provided.

            Be especially careful to distinguish between:
            - How often a benefit is given (frequency), and
            - When or how the benefit value changes (amount or eligibility criteria).

            If the context mentions that a benefit increases after a certain time, do not assume that means the benefit is only given at those intervals. 
            Clarify if the timing of the benefit is mentioned separately.

            If the document does not specify something (such as timing, frequency, or exceptions), do not guess or assume. 
            Just say the document does not provide that information.

            Do not invent new details, scenarios, or policy variations unless they are explicitly stated in the document.

            Your response must be at least 3 to 5 sentences long.

            After your response, include a confidence score (1–10) based on how well the context supports your answer.

            Format:
            Answer: <your answer here>

            Confidence Score: <number>/10 – <brief explanation>
            """)

        # Setup LLM chain with the custom prompt
        llm = Ollama(model=model_name, temperature=0.0, base_url=OLLAMA_API_BASE)
        llm_chain = LLMChain(llm=llm, prompt=custom_prompt)

        # Run LLM chain with appropriate variables
        if use_chat_history and conversation_history.strip():
            answer = llm_chain.run({
                "conversation_history": conversation_history,
                "context": context,
                "question": question
            })
        else:
            answer = llm_chain.run({
                "context": context,
                "question": question
            })

        # Return answer and source docs
        return {
            "answer": answer,
            "source_documents": relevant_docs
        }, None

    except Exception as e:
        return None, f"Error answering question: {str(e)}"

def process_user_query(message, conversation, model_name="llama3.2:latest", use_chat_history=True):
    """Process a user query and generate a response"""
    # Add user message to conversation with unique ID
    user_message = Message(
        id=generate_message_id(),
        role="user", 
        content=message
    )
    conversation.messages.append(user_message)
    
    # Find the best matching document
    top_docs, error = find_relevant_documents(message)
    if error:
        return None, error, [], use_chat_history, None, None
    
    if not top_docs:
        return None, "No relevant documents found.", [], use_chat_history, None, None
    
    # Get the best matching document
    best_doc = top_docs[0]
    
    # Load embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDINGS_MODEL,
        model_kwargs={"device": "cpu"}
    )
    
    # Load FAISS index for the best document
    index_name = best_doc["index_name"]
    vectorstore, error = load_faiss_index(index_name, embeddings)
    if error:
        return None, error, [], use_chat_history, None, None
    
    # Build conversation history context (will be empty string if use_chat_history is False)
    conversation_history = build_chat_history_context(
        conversation.messages, 
        max_history=5, 
        use_history=use_chat_history
    )
    
    # Get answer from the best document
    result, error = answer_question(
        vectorstore, 
        message, 
        conversation_history, 
        model_name, 
        use_chat_history
    )
    if error:
        return None, error, [], use_chat_history, None, None
    
    # Extract page numbers from source documents
    page_numbers = []
    for src_doc in result["source_documents"]:
        if src_doc.metadata.get("page", 0) not in page_numbers:
            page_numbers.append(src_doc.metadata.get("page", 0))
    
    # Sort page numbers
    page_numbers.sort()
    
    # Add assistant message to conversation with unique ID
    assistant_message = Message(
        id=generate_message_id(),
        role="assistant", 
        content=result["answer"]
    )
    conversation.messages.append(assistant_message)
    
    return {
        "answer": result["answer"],
        "document": best_doc["original_name"],
    }, None, page_numbers, use_chat_history, best_doc.get("match_type", "unknown"), best_doc.get("similarity", 0.0)

def cleanup_old_sessions():
    """Clean up sessions that have too many messages to prevent memory overflow"""
    MAX_MESSAGES_PER_SESSION = 100
    sessions_to_cleanup = []
    
    for session_id, conversation in session_conversations.items():
        if len(conversation.messages) > MAX_MESSAGES_PER_SESSION:
            # Keep only the most recent messages
            conversation.messages = conversation.messages[-50:]  # Keep last 50 messages
    
    print(f"Cleaned up sessions. Total active sessions: {len(session_conversations)}")

@app.get("/")
async def root():
    return {
        "name": "DocumentMind Enhanced",
        "description": "Your AI-powered document chat assistant with fuzzy matching",
        "version": "2.1.0",
        "features": [
            "Fuzzy string matching for typos",
            "Multi-level document ranking",
            "Semantic similarity fallback",
            "Enhanced keyword matching"
        ]
    }

@app.get("/status")
async def check_status():
    ollama_running, models = check_ollama_status()
    return {
        "service": "DocumentMind Enhanced",
        "status": "online",
        "ollama_running": ollama_running,
        "available_models": models if ollama_running else [],
        "faiss_indexes_path_exists": os.path.exists(FAISS_INDEX_DIR),
        "metadata_file_exists": os.path.exists(METADATA_FILE),
        "active_sessions": len(session_conversations),
        "fuzzy_matching_enabled": True,
        "supported_libraries": ["fuzzywuzzy", "python-levenshtein", "sentence-transformers"]
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest, 
    background_tasks: BackgroundTasks
):
    # Run cleanup task in background occasionally
    if len(session_conversations) > 50:  # Run cleanup when we have many sessions
        background_tasks.add_task(cleanup_old_sessions)
    
    # Check Ollama server status and available models
    ollama_running, available_models = check_ollama_status()
    if not ollama_running:
        raise HTTPException(status_code=503, detail="Language model server is not running")
    
    if request.model_name not in available_models:
        return ChatResponse(
            answer="Requested model is not available.",
            page_numbers=[],
            session_id="error",
            success=False,
            error=f"Model '{request.model_name}' not available. Available models: {', '.join(available_models)}",
            used_chat_history=request.use_chat_history
        )
    
    # Check if user actually sent a message
    if not request.message.strip():
        return ChatResponse(
            answer="No message received. Please send a valid message.",
            page_numbers=[],
            session_id=request.session_id or generate_session_id(),
            success=False,
            error="Empty message",
            used_chat_history=request.use_chat_history
        )
    
    # Get or create session-based conversation
    conversation = get_or_create_session(request.session_id)
    
    # Process user query and get response
    result, error, page_numbers, history_used, match_type, similarity_score = process_user_query(
        request.message,
        conversation,
        request.model_name,
        request.use_chat_history  # Pass the toggle parameter
    )
    
    if error:
        return ChatResponse(
            answer="Sorry, I couldn't find an answer to your question. Please try rephrasing or ask something else.",
            page_numbers=[],
            session_id=conversation.session_id,
            success=False,
            error=error,
            used_chat_history=history_used
        )
    
    # Return success response with session ID and enhanced metadata
    return ChatResponse(
        answer=result["answer"],
        page_numbers=page_numbers,
        document_name=result["document"],
        session_id=conversation.session_id,
        success=True,
        error=None,
        used_chat_history=history_used,
        match_type=match_type,
        similarity_score=similarity_score
    )

@app.get("/conversations/{session_id}", response_model=List[Message])
async def get_conversation(session_id: str):
    """Get conversation history for a specific session"""
    if session_id not in session_conversations:
        return []
    
    return session_conversations[session_id].messages

@app.delete("/conversations/{session_id}")
async def clear_conversation(session_id: str):
    """Clear conversation history for a specific session"""
    if session_id in session_conversations:
        session_conversations[session_id].messages = []
        return {"message": "Conversation cleared successfully"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@app.get("/sessions")
async def get_active_sessions():
    """Get list of active session IDs and their message counts"""
    sessions_info = {}
    for session_id, conversation in session_conversations.items():
        sessions_info[session_id] = {
            "message_count": len(conversation.messages),
            "has_messages": len(conversation.messages) > 0
        }
    return sessions_info

@app.get("/test-fuzzy/{query}")
async def test_fuzzy_matching_endpoint(query: str):
    """Test endpoint to check fuzzy matching results"""
    try:
        results, error = find_relevant_documents(query, top_k=3)
        
        if error:
            return {"error": error}
        
        if not results:
            return {"message": "No matches found", "query": query}
        
        return {
            "query": query,
            "results": [
                {
                    "document": result["original_name"],
                    "similarity": result["similarity"],
                    "match_type": result.get("match_type", "unknown"),
                    "matched_term": result.get("matched_term", "N/A")
                }
                for result in results
            ]
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/document-keywords")
async def get_document_keywords():
    """Get the current document keyword mappings"""
    return {
        "document_keywords": DOCUMENT_KEYWORDS,
        "total_keywords": len(DOCUMENT_KEYWORDS),
        "description": "Keywords mapped to document indices for direct matching"
    }

@app.get("/documents")
async def list_documents():
    """List all available documents with their metadata"""
    metadata_path = os.path.join(FAISS_INDEX_DIR, "document_embeddings")
    
    if not os.path.exists(metadata_path):
        return {
            "documents": [],
            "total_count": 0,
            "error": "No document embeddings found"
        }
    
    try:
        documents = []
        for file in os.listdir(metadata_path):
            if file.endswith(".pkl"):
                file_path = os.path.join(metadata_path, file)
                with open(file_path, "rb") as f:
                    doc_data = pickle.load(f)
                    documents.append({
                        "index_name": doc_data["index_name"],
                        "original_name": doc_data["original_name"],
                        "embedding_size": len(doc_data["embedding"]) if "embedding" in doc_data else 0
                    })
        
        return {
            "documents": documents,
            "total_count": len(documents),
            "faiss_index_path": FAISS_INDEX_DIR
        }
    except Exception as e:
        return {
            "documents": [],
            "total_count": 0,
            "error": f"Error loading document metadata: {str(e)}"
        }

@app.post("/sessions/{session_id}/export")
async def export_conversation(session_id: str):
    """Export conversation history as JSON"""
    if session_id not in session_conversations:
        raise HTTPException(status_code=404, detail="Session not found")
    
    conversation = session_conversations[session_id]
    
    # Convert to exportable format
    export_data = {
        "session_id": session_id,
        "export_timestamp": str(np.datetime64('now')),
        "message_count": len(conversation.messages),
        "messages": [
            {
                "id": msg.id,
                "role": msg.role,
                "content": msg.content,
                "timestamp": None  # Could add timestamps in future
            }
            for msg in conversation.messages
        ]
    }
    
    return export_data

@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint"""
    try:
        # Check Ollama connection
        ollama_running, available_models = check_ollama_status()
        
        # Check FAISS indices
        faiss_healthy = os.path.exists(FAISS_INDEX_DIR)
        
        # Check document embeddings
        embeddings_path = os.path.join(FAISS_INDEX_DIR, "document_embeddings")
        embeddings_healthy = os.path.exists(embeddings_path)
        
        # Count available documents
        doc_count = 0
        if embeddings_healthy:
            try:
                doc_count = len([f for f in os.listdir(embeddings_path) if f.endswith(".pkl")])
            except:
                doc_count = 0
        
        # Overall health status
        overall_healthy = all([
            ollama_running,
            faiss_healthy,
            embeddings_healthy,
            doc_count > 0
        ])
        
        return {
            "status": "healthy" if overall_healthy else "degraded",
            "timestamp": str(np.datetime64('now')),
            "components": {
                "ollama_server": {
                    "status": "up" if ollama_running else "down",
                    "available_models": available_models
                },
                "faiss_indices": {
                    "status": "available" if faiss_healthy else "missing",
                    "path": FAISS_INDEX_DIR
                },
                "document_embeddings": {
                    "status": "available" if embeddings_healthy else "missing",
                    "document_count": doc_count
                },
                "active_sessions": {
                    "count": len(session_conversations),
                    "status": "normal" if len(session_conversations) < 100 else "high_load"
                }
            },
            "features": {
                "fuzzy_matching": True,
                "chat_history": True,
                "multi_session": True,
                "background_cleanup": True
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "timestamp": str(np.datetime64('now')),
            "error": str(e)
        }

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    print("🚀 Starting DocumentMind Enhanced...")
    print(f"📁 FAISS Index Directory: {FAISS_INDEX_DIR}")
    print(f"🤖 Ollama API Base: {OLLAMA_API_BASE}")
    print(f"🧠 Embeddings Model: {EMBEDDINGS_MODEL}")
    
    # Check initial status
    ollama_running, models = check_ollama_status()
    print(f"🔍 Ollama Status: {'✅ Running' if ollama_running else '❌ Not Running'}")
    if ollama_running:
        print(f"📋 Available Models: {', '.join(models)}")
    
    # Check document indices
    if os.path.exists(FAISS_INDEX_DIR):
        print("📊 FAISS indices directory found")
        embeddings_path = os.path.join(FAISS_INDEX_DIR, "document_embeddings")
        if os.path.exists(embeddings_path):
            try:
                doc_count = len([f for f in os.listdir(embeddings_path) if f.endswith(".pkl")])
                print(f"📚 Found {doc_count} document embeddings")
            except:
                print("⚠️  Could not count document embeddings")
        else:
            print("⚠️  Document embeddings directory not found")
    else:
        print("❌ FAISS indices directory not found")
    
    print("✅ DocumentMind Enhanced is ready!")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on application shutdown"""
    print("🛑 Shutting down DocumentMind Enhanced...")
    
    # Clear session data
    session_conversations.clear()
    print("🧹 Cleared session data")
    
    print("👋 DocumentMind Enhanced shutdown complete")

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "document_rank_2:app",  # Assuming this file is named main.py
        host="0.0.0.0",
        port=8005,
        reload=True,
        log_level="info"
    )