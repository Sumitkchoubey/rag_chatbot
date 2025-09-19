import React, { useState, useEffect, useRef } from 'react';
import './DocumentMind.css';

// Icons fallback setup (using lucide-react if available)
let MessageSquare, Send, RefreshCw, Info, AlertTriangle, FileText, Trash2, Plus, Star, Download;

try {
  const Lucide = require('lucide-react');
  MessageSquare = Lucide.MessageSquare;
  Send = Lucide.Send;
  RefreshCw = Lucide.RefreshCw;
  Info = Lucide.Info;
  AlertTriangle = Lucide.AlertTriangle;
  FileText = Lucide.FileText;
  Trash2 = Lucide.Trash2;
  Plus = Lucide.Plus;
  Star = Lucide.Star;
  Download = Lucide.Download;
} catch (e) {
  const IconFallback = ({ className }) => <span className={className}>●</span>;
  MessageSquare = IconFallback;
  Send = () => <span>→</span>;
  RefreshCw = () => <span>⟳</span>;
  Info = () => <span>ℹ</span>;
  AlertTriangle = () => <span>⚠</span>;
  FileText = () => <span>📄</span>;
  Trash2 = () => <span>🗑️</span>;
  Plus = () => <span>+</span>;
  Star = () => <span>⭐</span>;
  Download = () => <span>📥</span>;
}

const DEMO_MODE = false;
const BACKEND_URL = 'http://localhost:8005';

// Generate a unique conversation ID
const generateConversationId = () => {
  return 'conv_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
};

// LocalStorage helpers for conversation management
const getStoredConversations = () => {
  try {
    const stored = localStorage.getItem('documentmind_conversations');
    return stored ? JSON.parse(stored) : {};
  } catch (e) {
    console.error('Error parsing stored conversations:', e);
    return {};
  }
};

const saveConversation = (conversationId, messages, title = null) => {
  try {
    const conversations = getStoredConversations();
    const conversationTitle = title || generateConversationTitle(messages);
    
    conversations[conversationId] = {
      id: conversationId,
      title: conversationTitle,
      messages: messages,
      lastUpdated: new Date().toISOString(),
      createdAt: conversations[conversationId]?.createdAt || new Date().toISOString()
    };
    
    localStorage.setItem('documentmind_conversations', JSON.stringify(conversations));
  } catch (e) {
    console.error('Error saving conversation:', e);
  }
};

const deleteStoredConversation = (conversationId) => {
  try {
    const conversations = getStoredConversations();
    delete conversations[conversationId];
    localStorage.setItem('documentmind_conversations', JSON.stringify(conversations));
  } catch (e) {
    console.error('Error deleting conversation:', e);
  }
};

const generateConversationTitle = (messages) => {
  const firstUserMessage = messages.find(msg => msg.role === 'user');
  if (!firstUserMessage) return 'New Conversation';
  
  const words = firstUserMessage.content.split(' ');
  const title = words.slice(0, 6).join(' ');
  return title.length > 40 ? title.substring(0, 37) + '...' : title;
};

// Save rating to backend only
const saveRatingToBackend = async (rating, question, answer, timestamp, conversationId) => {
  try {
    const response = await fetch(`${BACKEND_URL}/save-rating`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        rating,
        question,
        answer,
        timestamp,
        conversationId
      }),
    });

    if (response.ok) {
      console.log('Rating saved to backend successfully');
      return { success: true };
    } else {
      console.error('Failed to save rating to backend');
      return { success: false, error: 'Failed to save' };
    }
  } catch (e) {
    console.error('Error saving rating to backend:', e);
    return { success: false, error: 'Network error while saving rating' };
  }
};

// Star Rating Component with feedback
const StarRating = ({ messageId, onRate, currentRating = 0, feedback = null }) => {
  const [hoveredRating, setHoveredRating] = useState(0);
  const [selectedRating, setSelectedRating] = useState(currentRating);

  const handleStarClick = (rating) => {
    setSelectedRating(rating);
    onRate(messageId, rating);
  };

  return (
    <div className="star-rating">
      <span className="rating-label">Rate this response:</span>
      <div className="stars">
        {[1, 2, 3, 4, 5].map((star) => (
          <button
            key={star}
            className={`star ${star <= (hoveredRating || selectedRating) ? 'star-filled' : 'star-empty'}`}
            onClick={() => handleStarClick(star)}
            onMouseEnter={() => setHoveredRating(star)}
            onMouseLeave={() => setHoveredRating(0)}
            title={`Rate ${star} star${star !== 1 ? 's' : ''}`}
            disabled={feedback && feedback.success}
          >
            <Star size={16} />
          </button>
        ))}
      </div>
      {selectedRating > 0 && !feedback && (
        <span className="rating-text">({selectedRating}/5)</span>
      )}
      {feedback && (
        <div className={`rating-feedback ${feedback.success ? 'rating-feedback-success' : 'rating-feedback-error'}`}>
          {feedback.success ? (
            <span className="feedback-success">✓ Thanks for your feedback!</span>
          ) : (
            <span className="feedback-error">✗ {feedback.error}</span>
          )}
        </div>
      )}
    </div>
  );
};

export default function DocumentMindChat() {
  const [messages, setMessages] = useState([
    { 
      id: 'welcome_msg', 
      role: 'assistant', 
      content: "Hello! I'm DocumentMind. Ask me anything about your documents and I'll do my best to help you.",
      rating: null,
      ratingFeedback: null
    }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [status, setStatus] = useState({ isOnline: DEMO_MODE, models: DEMO_MODE ? ['llama3.2-demo'] : [] });
  const [currentConversationId, setCurrentConversationId] = useState(null);
  const [conversationHistory, setConversationHistory] = useState({});
  const [viewingPageNumbers, setViewingPageNumbers] = useState([]);
  const [currentDocument, setCurrentDocument] = useState(null);

  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    // Load conversation history from localStorage
    const storedConversations = getStoredConversations();
    setConversationHistory(storedConversations);
    
    if (!DEMO_MODE) {
      checkStatus();
    }
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    // Save current conversation whenever messages change (except welcome message)
    if (currentConversationId && messages.length > 1) {
      saveConversation(currentConversationId, messages);
      // Update conversation history state
      const updatedConversations = getStoredConversations();
      setConversationHistory(updatedConversations);
    }
  }, [messages, currentConversationId]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const checkStatus = async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/status`);
      const data = await response.json();
      setStatus({
        isOnline: data.ollama_running,
        models: data.available_models
      });
    } catch (err) {
      setError('Unable to connect to server');
      setStatus({ isOnline: false, models: [] });
    }
  };

  const loadConversation = (conversationId) => {
    const conversations = getStoredConversations();
    const conversation = conversations[conversationId];
    
    if (conversation) {
      setMessages(conversation.messages);
      setCurrentConversationId(conversationId);
      setError(null);
      setViewingPageNumbers([]);
      setCurrentDocument(null);
    }
  };

  const handleRating = async (messageId, rating) => {
    // Update message with rating (show loading state)
    setMessages(prevMessages => 
      prevMessages.map(msg => 
        msg.id === messageId ? { ...msg, rating, ratingFeedback: { loading: true } } : msg
      )
    );

    // Find the rated message and its corresponding question
    const ratedMessage = messages.find(msg => msg.id === messageId);
    const messageIndex = messages.findIndex(msg => msg.id === messageId);
    const questionMessage = messageIndex > 0 ? messages[messageIndex - 1] : null;

    if (ratedMessage && questionMessage && questionMessage.role === 'user') {
      const timestamp = new Date().toISOString();
      
      let result = { success: false, error: 'Unknown error' };
      
      if (!DEMO_MODE) {
        // Save to backend
        result = await saveRatingToBackend(
          rating,
          questionMessage.content,
          ratedMessage.content,
          timestamp,
          currentConversationId
        );
      } else {
        // Demo mode - simulate success
        await new Promise(resolve => setTimeout(resolve, 1000));
        result = { success: true };
      }

      // Update message with feedback
      setMessages(prevMessages => 
        prevMessages.map(msg => 
          msg.id === messageId ? { 
            ...msg, 
            rating, 
            ratingFeedback: result
          } : msg
        )
      );

      // Clear feedback after 3 seconds
      setTimeout(() => {
        setMessages(prevMessages => 
          prevMessages.map(msg => 
            msg.id === messageId ? { ...msg, ratingFeedback: null } : msg
          )
        );
      }, 3000);
    }
  };

  const sendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userMessage = input.trim();
    setInput('');
    
    // If this is a new conversation (no current ID), create one
    let conversationId = currentConversationId;
    if (!conversationId) {
      conversationId = generateConversationId();
      setCurrentConversationId(conversationId);
    }
    
    // Create user message with unique ID
    const tempUserMessage = { 
      id: 'user_' + Date.now(), 
      role: 'user', 
      content: userMessage 
    };
    
    const updatedMessages = [...messages, tempUserMessage];
    setMessages(updatedMessages);
    setLoading(true);
    setError(null);

    if (DEMO_MODE) {
      setTimeout(() => {
        const demoResponse = { 
          id: 'demo_' + Date.now(), 
          role: 'assistant', 
          content: "This is a demo response. Your conversation is being saved locally.",
          rating: null,
          ratingFeedback: null
        };
        const finalMessages = [...updatedMessages, demoResponse];
        setMessages(finalMessages);
        setLoading(false);
      }, 1500);
      return;
    }

    try {
      const response = await fetch(`${BACKEND_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: userMessage,
          model_name: status.models.includes('llama3.2') ? 'llama3.2' : status.models[0],
          session_id: conversationId // Use conversation ID as session ID
        }),
      });

      const data = await response.json();

      if (data.success) {
        // Add assistant response with unique ID and rating field
        const assistantMessage = { 
          id: 'assistant_' + Date.now(), 
          role: 'assistant', 
          content: data.answer,
          rating: null,
          ratingFeedback: null
        };
        
        const finalMessages = [...updatedMessages, assistantMessage];
        setMessages(finalMessages);
        setViewingPageNumbers(data.page_numbers || []);
        setCurrentDocument(data.document_name || null);
        
        // Save conversation immediately after getting response
        saveConversation(conversationId, finalMessages);
        setConversationHistory(getStoredConversations());
      } else {
        setError(data.error || 'Error processing your question');
      }
    } catch (err) {
      setError('Network error. Please try again.');
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  };

  const startNewConversation = () => {
    setMessages([{ 
      id: 'welcome_msg', 
      role: 'assistant', 
      content: "Hello! I'm DocumentMind. Ask me anything about your documents and I'll do my best to help you.",
      rating: null,
      ratingFeedback: null
    }]);
    setCurrentConversationId(null);
    setViewingPageNumbers([]);
    setCurrentDocument(null);
    setError(null);
  };

  const deleteConversation = (conversationId, e) => {
    e.stopPropagation();
    
    // Delete from localStorage
    deleteStoredConversation(conversationId);
    
    // Update state
    setConversationHistory(getStoredConversations());
    
    // If we're currently viewing this conversation, start a new one
    if (currentConversationId === conversationId) {
      startNewConversation();
    }
  };

  const formatMessage = (content) => content.split('\n').map((line, i) => (
    <React.Fragment key={i}>{line}<br /></React.Fragment>
  ));

  // Sort conversations by last updated (most recent first)
  const sortedConversations = Object.values(conversationHistory)
    .sort((a, b) => new Date(b.lastUpdated) - new Date(a.lastUpdated));

  const getCurrentConversationTitle = () => {
    if (!currentConversationId) return 'New Conversation';
    const conversation = conversationHistory[currentConversationId];
    return conversation ? conversation.title : 'Current Conversation';
  };

  return (
    <div className="document-mind">
      {/* Header */}
      <div className="document-mind-header">
        <div className="header-content">
          <div className="app-title">
            <MessageSquare className="icon" />
            <h1>DocumentMind</h1>
          </div>
          <div className="header-controls">
            <span className="current-conversation-title">
              {getCurrentConversationTitle()}
            </span>
            {status.isOnline ? (
              <span className="status-indicator status-online">
                <span className="status-dot"></span>Online
              </span>
            ) : (
              <span className="status-indicator status-offline">
                <span className="status-dot"></span>Offline
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="main-content">
        <div className="sidebar">
          <button onClick={startNewConversation} className="new-conversation-btn">
            <Plus size={16} />
            New Conversation
          </button>
          
          <div className="conversations-section">
            <h3 className="conversations-title">RECENT CONVERSATIONS</h3>
            <div className="conversations-list">
              {sortedConversations.length > 0 ? (
                sortedConversations.map((conversation) => (
                  <div key={conversation.id} className="conversation-item-wrapper">
                    <button
                      onClick={() => loadConversation(conversation.id)}
                      className={`conversation-item ${currentConversationId === conversation.id ? 'conversation-active' : ''}`}
                      title={conversation.title}
                    >
                      <div className="conversation-title">{conversation.title}</div>
                      <div className="conversation-date">
                        {new Date(conversation.lastUpdated).toLocaleDateString()}
                      </div>
                    </button>
                    <button 
                      onClick={(e) => deleteConversation(conversation.id, e)} 
                      className="delete-conversation-btn"
                      title="Delete conversation"
                    >
                      <Trash2 size={14} />
                    </button>
                  </div>
                ))
              ) : (
                <div className="no-conversations">
                  <p>No conversations yet</p>
                  <p>Start chatting to see your history here</p>
                </div>
              )}
            </div>
          </div>
          
          <div className="storage-info">
            <p className="storage-note">
              💾 Conversations saved locally in your browser
            </p>
          </div>
        </div>

        {/* Chat Area */}
        <div className="chat-area">
          {currentDocument && (
            <div className="document-info">
              <FileText className="icon" />
              <span className="document-name">{currentDocument}</span>
              {viewingPageNumbers.length > 0 && (
                <span className="page-numbers">(Pages: {viewingPageNumbers.join(', ')})</span>
              )}
            </div>
          )}

          <div className="messages-container">
            {messages.map((message) => (
              <div key={message.id} className={`message-row ${message.role === 'user' ? 'message-user' : 'message-assistant'}`}>
                <div className={`message-content ${message.role === 'user' ? 'message-content-user' : 'message-content-assistant'}`}>
                  {formatMessage(message.content)}
                  {message.role === 'assistant' && message.id !== 'welcome_msg' && (
                    <StarRating 
                      messageId={message.id}
                      onRate={handleRating}
                      currentRating={message.rating || 0}
                      feedback={message.ratingFeedback}
                    />
                  )}
                </div>
              </div>
            ))}
            {loading && (
              <div className="loading-indicator">
                <div className="loading-bubble">
                  <RefreshCw className="loading-icon" />
                  <span>DocumentMind is thinking...</span>
                </div>
              </div>
            )}
            {error && (
              <div className="error-message">
                <div className="error-content">
                  <AlertTriangle className="error-icon" />
                  {error}
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input Area */}
          <div className="input-area">
            <form onSubmit={sendMessage} className="input-form">
              <input
                ref={inputRef}
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask anything about your documents..."
                className="message-input"
                disabled={!status.isOnline || loading}
              />
              <button
                type="submit"
                className={`send-button ${status.isOnline && input.trim() && !loading ? 'send-button-active' : 'send-button-disabled'}`}
                disabled={!status.isOnline || !input.trim() || loading}
              >
                <Send />
              </button>
            </form>
            {!status.isOnline && (
              <div className="status-warning">
                <Info className="status-warning-icon" />
                DocumentMind is currently offline. Please check your server connection.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}