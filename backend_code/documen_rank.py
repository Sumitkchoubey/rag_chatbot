import os
import numpy as np
import pickle
import re
from typing import List, Dict, Any, Optional
import requests
import json
import uvicorn
import uuid
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
import os
import pickle
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
 
def find_relevant_documents(query, top_k=5):
    """Find the most relevant documents for a query with improved keyword prioritization and misspelling handling"""
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
    
    # Initialize embeddings for similarity calculations
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDINGS_MODEL,
        model_kwargs={"device": "cpu"}
    )
    
    query_lower = query.lower()
    query_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', query_lower))
    
    print(f"Processing query: '{query}' with words: {query_words}")
    
    # STEP 1: Check predefined keyword mapping with enhanced misspelling detection
    best_keyword_match = None
    best_keyword_score = 0.0
    
    for keyword, doc_index in DOCUMENT_KEYWORDS.items():
        # Direct keyword match
        if keyword in query_lower:
            for doc in docs_data:
                if doc["index_name"] == doc_index:
                    print(f"✓ Direct keyword match: '{keyword}' -> {doc['original_name']}")
                    return [{
                        "index_name": doc["index_name"],
                        "original_name": doc["original_name"],
                        "similarity": 1.0,
                        "match_type": "direct_keyword"
                    }], None
        
        # Enhanced misspelling detection for keywords
        for query_word in query_words:
            # Check similarity between query word and predefined keyword
            similarity = enhanced_string_similarity(query_word, keyword)
            
            if similarity > 0.6 and similarity > best_keyword_score:  # Lower threshold for better detection
                best_keyword_match = {
                    'keyword': keyword,
                    'doc_index': doc_index,
                    'query_word': query_word,
                    'similarity': similarity
                }
                best_keyword_score = similarity
    
    # Return best keyword match if found
    if best_keyword_match:
        for doc in docs_data:
            if doc["index_name"] == best_keyword_match['doc_index']:
                print(f"✓ Keyword misspelling match: '{best_keyword_match['query_word']}' -> '{best_keyword_match['keyword']}' -> {doc['original_name']} (similarity: {best_keyword_match['similarity']:.3f})")
                return [{
                    "index_name": doc["index_name"],
                    "original_name": doc["original_name"],
                    "similarity": 0.95,  # High score but not 1.0 to indicate it's a misspelling match
                    "match_type": "keyword_misspelling"
                }], None
    
    # STEP 2: Check direct filename word matching
    filename_matches = []
    for doc in docs_data:
        filename_lower = doc["original_name"].lower()
        filename_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', filename_lower))
        
        # Check for direct word matches
        matches = query_words.intersection(filename_words)
        if matches:
            match_strength = len(matches) / len(query_words) if query_words else 0
            
            if match_strength >= 0.3:  # Lower threshold for better matching
                filename_matches.append({
                    "doc": doc,
                    "matches": matches,
                    "strength": match_strength
                })
    
    # If we have filename matches, return the best one
    if filename_matches:
        best_filename_match = max(filename_matches, key=lambda x: x["strength"])
        doc = best_filename_match["doc"]
        print(f"✓ Filename word match: {best_filename_match['matches']} -> {doc['original_name']} (strength: {best_filename_match['strength']:.2f})")
        return [{
            "index_name": doc["index_name"],
            "original_name": doc["original_name"],
            "similarity": 0.9,  # High score for filename matches
            "match_type": "filename_direct"
        }], None
    
    # STEP 3: Enhanced filename misspelling detection
    filename_misspelling_matches = []
    
    for doc in docs_data:
        doc_name_words = re.findall(r'\b[a-zA-Z]{3,}\b', doc["original_name"].lower())
        
        for query_word in query_words:
            for doc_word in doc_name_words:
                similarity = enhanced_string_similarity(query_word, doc_word)
                
                if similarity > 0.6:  # Lower threshold for better misspelling detection
                    filename_misspelling_matches.append({
                        'query_word': query_word,
                        'matched_word': doc_word,
                        'similarity': similarity,
                        'doc': doc
                    })
    
    # Return best filename misspelling match if found
    if filename_misspelling_matches:
        best_match = max(filename_misspelling_matches, key=lambda x: x['similarity'])
        doc = best_match['doc']
        print(f"✓ Filename misspelling match: '{best_match['query_word']}' -> '{best_match['matched_word']}' -> {doc['original_name']} (similarity: {best_match['similarity']:.3f})")
        return [{
            "index_name": doc["index_name"],
            "original_name": doc["original_name"],
            "similarity": 0.85,  # Good score for misspelling matches
            "match_type": "filename_misspelling"
        }], None
    
    # STEP 4: Semantic similarity fallback (only if no other matches found)
    print(f"No direct/misspelling matches found for '{query}', using semantic similarity...")
    
    query_embedding = embeddings.embed_query(query)
    
    # Calculate similarity with each document
    docs_with_scores = []
    for doc in docs_data:
        query_emb_2d = np.array(query_embedding).reshape(1, -1)
        doc_emb_2d = np.array(doc["embedding"]).reshape(1, -1)
        
        similarity = cosine_similarity(query_emb_2d, doc_emb_2d)[0][0]
        docs_with_scores.append({
            "index_name": doc["index_name"],
            "original_name": doc["original_name"],
            "similarity": float(similarity),
            "match_type": "semantic"
        })
    
    print(f"Semantic similarity results for '{query}':")
    for doc in docs_with_scores:
        print(f"  {doc['original_name']}: {doc['similarity']:.3f}")
    
    # Sort by similarity (highest first)
    docs_with_scores.sort(key=lambda x: x["similarity"], reverse=True)
    
    return docs_with_scores[:top_k], None


def enhanced_string_similarity(s1, s2):
    """Enhanced string similarity that combines multiple methods for better misspelling detection"""
    s1, s2 = s1.lower(), s2.lower()
    
    # Method 1: SequenceMatcher ratio (good for general similarity)
    seq_ratio = SequenceMatcher(None, s1, s2).ratio()
    
    # Method 2: Levenshtein-based similarity (good for typos)
    def levenshtein_similarity(s1, s2):
        if len(s1) == 0 or len(s2) == 0:
            return 0.0
        
        # Calculate Levenshtein distance
        d = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]
        
        for i in range(len(s1) + 1):
            d[i][0] = i
        for j in range(len(s2) + 1):
            d[0][j] = j
            
        for i in range(1, len(s1) + 1):
            for j in range(1, len(s2) + 1):
                cost = 0 if s1[i-1] == s2[j-1] else 1
                d[i][j] = min(
                    d[i-1][j] + 1,      # deletion
                    d[i][j-1] + 1,      # insertion
                    d[i-1][j-1] + cost  # substitution
                )
        
        max_len = max(len(s1), len(s2))
        return (max_len - d[len(s1)][len(s2)]) / max_len
    
    levenshtein_ratio = levenshtein_similarity(s1, s2)
    
    # Method 3: Common prefix/suffix bonus
    prefix_bonus = 0
    suffix_bonus = 0
    
    # Check common prefix
    min_len = min(len(s1), len(s2))
    for i in range(min_len):
        if s1[i] == s2[i]:
            prefix_bonus += 0.1
        else:
            break
    
    # Check common suffix  
    for i in range(1, min_len + 1):
        if s1[-i] == s2[-i]:
            suffix_bonus += 0.05
        else:
            break
    
    # Combine all methods with weights
    final_score = (
        seq_ratio * 0.4 +           # 40% weight for sequence matcher
        levenshtein_ratio * 0.5 +   # 50% weight for Levenshtein
        min(prefix_bonus, 0.1) +    # Up to 10% bonus for common prefix
        min(suffix_bonus, 0.05)     # Up to 5% bonus for common suffix
    )
    
    return min(final_score, 1.0)  # Cap at 1.0


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
        print(context)
        
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
        return None, error, [], use_chat_history
    
    if not top_docs:
        return None, "No relevant documents found.", [], use_chat_history
    
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
        return None, error, [], use_chat_history
    
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
        return None, error, [], use_chat_history
    
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
    }, None, page_numbers, use_chat_history

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
        "name": "DocumentMind",
        "description": "Your AI-powered document chat assistant",
        "version": "2.0.0"
    }

@app.get("/status")
async def check_status():
    ollama_running, models = check_ollama_status()
    return {
        "service": "DocumentMind",
        "status": "online",
        "ollama_running": ollama_running,
        "available_models": models if ollama_running else [],
        "faiss_indexes_path_exists": os.path.exists(FAISS_INDEX_DIR),
        "metadata_file_exists": os.path.exists(METADATA_FILE),
        "active_sessions": len(session_conversations)
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
    result, error, page_numbers, history_used = process_user_query(
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
    
    # Return success response with session ID
    return ChatResponse(
        answer=result["answer"],
        page_numbers=page_numbers,
        document_name=result["document"],
        session_id=conversation.session_id,
        success=True,
        error=None,
        used_chat_history=history_used
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

if __name__ == "__main__":
    uvicorn.run("rag_chat_app:app", host="0.0.0.0", port=8005, reload=True)