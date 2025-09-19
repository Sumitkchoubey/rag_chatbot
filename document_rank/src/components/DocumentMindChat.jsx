import React, { useState, useEffect, useRef } from 'react';
import { MessageSquare, Send, RefreshCw, Info, AlertTriangle, FileText } from 'lucide-react';

export default function DocumentMindChat() {
  // State management
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Hello! I\'m DocumentMind. Ask me anything about your documents and I\'ll do my best to help you.' }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [status, setStatus] = useState({ isOnline: false, models: [] });
  const [conversationId, setConversationId] = useState(null);
  const [availableConversations, setAvailableConversations] = useState([]);
  const [viewingPageNumbers, setViewingPageNumbers] = useState([]);
  const [currentDocument, setCurrentDocument] = useState(null);
  
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // Check server status on component mount
  useEffect(() => {
    checkStatus();
    fetchConversations();
  }, []);

  // Scroll to bottom of messages when messages change
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const checkStatus = async () => {
    try {
      const response = await fetch('/status');
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

  const fetchConversations = async () => {
    try {
      const response = await fetch('/conversations');
      const data = await response.json();
      setAvailableConversations(data);
    } catch (err) {
      console.error('Error fetching conversations:', err);
    }
  };

  const loadConversation = async (id) => {
    try {
      setLoading(true);
      const response = await fetch(`/conversation/${id}`);
      const data = await response.json();
      setMessages(data);
      setConversationId(id);
      setLoading(false);
    } catch (err) {
      setError('Error loading conversation');
      setLoading(false);
    }
  };

  const sendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;
    
    const userMessage = input.trim();
    setInput('');
    
    // Add user message to chat
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    
    // Show loading state
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: userMessage,
          model_name: status.models.includes('llama3.2') ? 'llama3.2' : status.models[0],
          conversation_id: conversationId
        }),
      });
      
      const data = await response.json();
      
      if (data.success) {
        // Update the conversation ID if it's new
        if (!conversationId) {
          setConversationId(data.conversation_id);
        }
        
        // Add assistant message to chat
        setMessages(prev => [...prev, { role: 'assistant', content: data.answer }]);
        
        // Store page numbers and document information
        setViewingPageNumbers(data.page_numbers || []);
        setCurrentDocument(data.document_name || null);
      } else {
        setError(data.error || 'Error processing your question');
      }
    } catch (err) {
      setError('Network error. Please try again.');
    } finally {
      setLoading(false);
      // Focus the input field again
      inputRef.current?.focus();
    }
  };

  const startNewConversation = () => {
    setMessages([{ role: 'assistant', content: 'Hello! I\'m DocumentMind. Ask me anything about your documents and I\'ll do my best to help you.' }]);
    setConversationId(null);
    setViewingPageNumbers([]);
    setCurrentDocument(null);
    setError(null);
  };

  const formatMessage = (content) => {
    // Simple formatting for messages
    return content.split('\n').map((line, i) => (
      <React.Fragment key={i}>
        {line}
        <br />
      </React.Fragment>
    ));
  };

  return (
    <div className="flex flex-col h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-blue-600 text-white py-4 px-6 shadow-md">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <MessageSquare className="mr-2" />
            <h1 className="text-xl font-bold">DocumentMind</h1>
          </div>
          <div className="flex items-center">
            {status.isOnline ? (
              <span className="text-green-300 text-sm flex items-center">
                <span className="w-2 h-2 bg-green-300 rounded-full mr-2"></span>
                Online
              </span>
            ) : (
              <span className="text-red-300 text-sm flex items-center">
                <span className="w-2 h-2 bg-red-300 rounded-full mr-2"></span>
                Offline
              </span>
            )}
          </div>
        </div>
      </div>
      
      {/* Main content area */}
      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <div className="w-64 bg-gray-100 border-r border-gray-200 p-4 hidden md:block">
          <div className="mb-6">
            <button 
              onClick={startNewConversation}
              className="w-full py-2 px-4 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition"
            >
              New Conversation
            </button>
          </div>
          
          <h3 className="text-sm font-semibold text-gray-500 mb-2">RECENT CONVERSATIONS</h3>
          <div className="space-y-1 max-h-96 overflow-y-auto">
            {availableConversations.map((id) => (
              <button
                key={id}
                onClick={() => loadConversation(id)}
                className={`w-full text-left py-2 px-3 rounded-md text-sm ${
                  conversationId === id ? 'bg-blue-100 text-blue-700' : 'hover:bg-gray-200'
                }`}
              >
                {id.split('_')[1] ? new Date(id.split('_')[1].substring(0, 8)).toLocaleDateString() : id}
              </button>
            ))}
          </div>
        </div>
        
        {/* Chat area */}
        <div className="flex-1 flex flex-col">
          {/* Document info bar */}
          {currentDocument && (
            <div className="bg-gray-100 p-2 border-b flex items-center text-sm">
              <FileText size={16} className="mr-2 text-gray-500" />
              <span className="font-medium mr-2">{currentDocument}</span>
              {viewingPageNumbers.length > 0 && (
                <span className="text-gray-500">
                  (Pages: {viewingPageNumbers.join(', ')})
                </span>
              )}
            </div>
          )}
          
          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {messages.map((message, index) => (
              <div 
                key={index} 
                className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div 
                  className={`max-w-3xl rounded-lg px-4 py-2 ${
                    message.role === 'user' 
                      ? 'bg-blue-600 text-white' 
                      : 'bg-white border border-gray-200'
                  }`}
                >
                  {formatMessage(message.content)}
                </div>
              </div>
            ))}
            {loading && (
              <div className="flex justify-start">
                <div className="max-w-3xl rounded-lg px-4 py-2 bg-white border border-gray-200">
                  <div className="flex items-center space-x-2">
                    <RefreshCw className="w-4 h-4 animate-spin text-gray-500" />
                    <span>DocumentMind is thinking...</span>
                  </div>
                </div>
              </div>
            )}
            {error && (
              <div className="flex justify-center">
                <div className="max-w-3xl rounded-lg px-4 py-2 bg-red-50 border border-red-200 text-red-700 flex items-center">
                  <AlertTriangle className="w-4 h-4 mr-2" />
                  {error}
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
          
          {/* Input area */}
          <div className="border-t border-gray-200 p-4">
            <form onSubmit={sendMessage} className="flex space-x-2">
              <input
                ref={inputRef}
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask anything about your documents..."
                className="flex-1 py-2 px-4 rounded-full border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                disabled={!status.isOnline || loading}
              />
              <button
                type="submit"
                className={`p-2 rounded-full ${
                  status.isOnline && input.trim() && !loading
                    ? 'bg-blue-600 text-white hover:bg-blue-700'
                    : 'bg-gray-200 text-gray-500 cursor-not-allowed'
                }`}
                disabled={!status.isOnline || !input.trim() || loading}
              >
                <Send className="w-5 h-5" />
              </button>
            </form>
            
            {/* System status indicator */}
            {!status.isOnline && (
              <div className="mt-2 text-xs text-red-600 flex items-center">
                <Info className="w-3 h-3 mr-1" />
                DocumentMind is currently offline. Please check your server connection.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}