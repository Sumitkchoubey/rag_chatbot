import os
import numpy as np
import pickle
import csv  # Add this missing import
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
OLLAMA_API_BASE = "http://43.204.205.229:11434"
RATINGS_CSV_FILE = "documentmind_ratings.csv"

# Models
class Message(BaseModel):
    id: str = Field(..., description="Unique message ID")
    role: str = Field(..., description="Role can be either 'user' or 'assistant'")
    content: str = Field(..., description="The content of the message")

class ChatRequest(BaseModel):
    message: str = Field(..., description="User's message/question")
    model_name: str = Field(default="llama3:latest", description="Model to use for response generation")
    session_id: Optional[str] = Field(default=None, description="Browser session ID")

class ChatResponse(BaseModel):
    answer: str = Field(..., description="Assistant's response")
    page_numbers: List[int] = Field(default_factory=list, description="Source page numbers")
    document_name: Optional[str] = Field(default=None, description="Source document name")
    session_id: str = Field(..., description="Browser session ID for tracking the chat")
    success: bool = Field(default=True, description="Whether the request was successful")
    error: Optional[str] = Field(default=None, description="Error message if any")

class ConversationData(BaseModel):
    session_id: str
    messages: List[Message] = Field(default_factory=list)
    document_context: Optional[str] = Field(default=None)

# Rating system models
class RatingRequest(BaseModel):
    rating: int = Field(..., ge=1, le=5, description="Rating from 1 to 5 stars")
    question: str = Field(..., description="The user's question that was rated")
    answer: str = Field(..., description="The assistant's answer that was rated")
    timestamp: str = Field(..., description="ISO timestamp when rating was given")
    conversationId: str = Field(..., description="ID of the conversation containing the rated exchange")

class RatingResponse(BaseModel):
    success: bool = Field(..., description="Whether the rating was saved successfully")
    message: str = Field(..., description="Response message")
    error: Optional[str] = Field(default=None, description="Error message if any")

class RatingStats(BaseModel):
    total_ratings: int = Field(..., description="Total number of ratings received")
    average_rating: float = Field(..., description="Average rating score")
    rating_distribution: Dict[int, int] = Field(..., description="Count of each rating (1-5 stars)")

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

def find_relevant_documents(query, top_k=5):
    """Find the most relevant documents for a query"""
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
    
    # Embed the query
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDINGS_MODEL,
        model_kwargs={"device": "cpu"}
    )
    query_embedding = embeddings.embed_query(query)
    
    # Calculate similarity with each document
    docs_with_scores = []
    for doc in docs_data:
        similarity = cosine_similarity(query_embedding, doc["embedding"])
        docs_with_scores.append({
            "index_name": doc["index_name"],
            "original_name": doc["original_name"],
            "similarity": similarity
        })
    
    print(docs_with_scores)
    # Sort by similarity (highest first)
    docs_with_scores.sort(key=lambda x: x["similarity"], reverse=True)
    
    # Return top k documents
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

def build_chat_history_context(messages, max_history=0):
    """Build context from chat history for the prompt"""
    # Only use the most recent messages to avoid context length issues
    recent_messages = messages[-max_history:] if len(messages) > max_history else messages
    
    history_text = ""
    for msg in recent_messages:
        role = "User" if msg.role == "user" else "Assistant"
        history_text += f"{role}: {msg.content}\n\n"
    
    return history_text

def answer_question(vectorstore, question, conversation_history="", model_name="llama3:latest"):
    """Answer a question using the RAG approach with Ollama, preserving the custom prompt"""
    try:
        # Create retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # Retrieve relevant documents
        relevant_docs = retriever.get_relevant_documents(question)

        # Combine document texts into one context string
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
         
        # Your existing custom prompt with placeholders
        custom_prompt = PromptTemplate.from_template("""
        You are DocumentMind, a helpful AI assistant that specializes in answering questions about documents.

        Always respond in a conversational, friendly manner as if you're chatting with the user.

        Previous conversation:
        {conversation_history}

        Context from documents:
        {context}

        User question:
        {question}
        You are answering questions based only on the provided context. Do not use any external knowledge or make assumptions beyond what the context explicitly states.

        Your tone should be comprehensive yet conversational.

        Distinguish carefully between:

        How often a benefit is given (frequency), and

        When or how the benefit value changes (amount or eligibility criteria).

        If the context says a benefit increases after a certain time, do not assume the benefit is only given at those intervals. Clarify timing only if it is mentioned separately.

        If the document does not specify something (such as timing, frequency, or exceptions), say so clearly. Do not guess or assume.

        Do not invent new details, scenarios, or policy variations unless they are explicitly stated in the document.

        If the answer can be found in the context, use the following format:

        Answer:
        <Comprehensive, conversational answer strictly grounded in the context. Clarify frequency, amount, or timing only if explicitly stated.>

        Confidence Score: <number>/10 – <Brief explanation of how well the context supports your answer.>

        If the answer cannot be found in the context, respond with EXACTLY this and nothing more:

        I am sorry unable to get answer
        """)

        # Setup LLM chain with the custom prompt
        llm = Ollama(model=model_name, temperature=0.0, base_url=OLLAMA_API_BASE)
        llm_chain = LLMChain(llm=llm, prompt=custom_prompt)

        # Run LLM chain with all variables filled
        answer = llm_chain.run({
            "conversation_history": conversation_history,
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

def save_rating_to_csv(rating_data: RatingRequest):
    """Save rating data to CSV file"""
    try:
        # Check if CSV file exists
        file_exists = os.path.isfile(RATINGS_CSV_FILE)
        
        # Open file in append mode
        with open(RATINGS_CSV_FILE, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            
            # Write header if file is new
            if not file_exists:
                writer.writerow(['rating', 'question', 'answer', 'timestamp', 'conversation_id'])
            
            # Write rating data
            writer.writerow([
                rating_data.rating,
                rating_data.question,
                rating_data.answer,
                rating_data.timestamp,
                rating_data.conversationId
            ])
        
        return True, "Rating saved successfully"
    
    except Exception as e:
        return False, f"Error saving rating: {str(e)}"

def get_rating_stats():
    """Get rating statistics from CSV file"""
    try:
        if not os.path.isfile(RATINGS_CSV_FILE):
            return RatingStats(
                total_ratings=0,
                average_rating=0.0,
                rating_distribution={1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
            )
        
        ratings = []
        with open(RATINGS_CSV_FILE, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                ratings.append(int(row['rating']))
        
        if not ratings:
            return RatingStats(
                total_ratings=0,
                average_rating=0.0,
                rating_distribution={1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
            )
        
        # Calculate statistics
        total_ratings = len(ratings)
        average_rating = sum(ratings) / total_ratings
        
        # Count distribution
        rating_distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for rating in ratings:
            rating_distribution[rating] += 1
        
        return RatingStats(
            total_ratings=total_ratings,
            average_rating=round(average_rating, 2),
            rating_distribution=rating_distribution
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting rating stats: {str(e)}")

def process_user_query(message, conversation, model_name="llama3:latest"):
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
        return None, error, []
    
    if not top_docs:
        return None, "No relevant documents found.", []
    
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
        return None, error, []
    
    # Build conversation history context
    #conversation_history = build_chat_history_context(conversation.messages)
    
    # Get answer from the best document
    result, error = answer_question(vectorstore, message, model_name)
    if error:
        return None, error, []
    
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
    }, None, page_numbers

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
            error=f"Model '{request.model_name}' not available. Available models: {', '.join(available_models)}"
        )
    
    # Check if user actually sent a message
    if not request.message.strip():
        return ChatResponse(
            answer="No message received. Please send a valid message.",
            page_numbers=[],
            session_id=request.session_id or generate_session_id(),
            success=False,
            error="Empty message"
        )
    
    # Get or create session-based conversation
    conversation = get_or_create_session(request.session_id)
    
    # Process user query and get response
    result, error, page_numbers = process_user_query(
        request.message,
        conversation,
        request.model_name
    )
    
    if error:
        return ChatResponse(
            answer="Sorry, I couldn't find an answer to your question. Please try rephrasing or ask something else.",
            page_numbers=[],
            session_id=conversation.session_id,
            success=False,
            error=error
        )
    
    # Return success response with session ID
    return ChatResponse(
        answer=result["answer"],
        page_numbers=page_numbers,
        document_name=result["document"],
        session_id=conversation.session_id,
        success=True,
        error=None
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

@app.post("/save-rating", response_model=RatingResponse)
async def submit_rating(rating_request: RatingRequest):
    """Submit a rating for a question-answer pair"""
    print(rating_request)
    try:

        success, message = save_rating_to_csv(rating_request)
        
        if success:
            return RatingResponse(success=True, message=message)
        else:
            return RatingResponse(success=False, message="Failed to save rating", error=message)
    
    except Exception as e:
        return RatingResponse(success=False, message="Internal server error", error=str(e))

@app.get("/rating-stats", response_model=RatingStats)
async def get_ratings_statistics():
    """Get rating statistics"""
    return get_rating_stats()

if __name__ == "__main__":
    uvicorn.run("back_test:app", host="0.0.0.0", port=8005, reload=True)