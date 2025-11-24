import { useState, useRef, useEffect } from 'react';
import { mockClassifyIntent, mockGetActivatedServers, type MCPServer } from '../services/mockApi';

interface Message {
  id: string;
  text: string;
  sender: 'user' | 'bot';
  timestamp: Date;
}

export default function ChatbotPage() {
  // Get authenticated user name from localStorage
  const authenticatedUser = localStorage.getItem('authenticatedUser') || 'there';
  
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      text: `Hello ${authenticatedUser}! How can I help you today?`,
      sender: 'bot',
      timestamp: new Date(),
    },
  ]);
  const [inputText, setInputText] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [activatedServers, setActivatedServers] = useState<MCPServer[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Auto-scroll to bottom when new messages are added
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!inputText.trim() || isProcessing) {
      return;
    }

    const userMessage: Message = {
      id: Date.now().toString(),
      text: inputText.trim(),
      sender: 'user',
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputText('');
    setIsProcessing(true);

    try {
      // Classify intent
      const intentResult = await mockClassifyIntent(userMessage.text);
      
      // Get activated servers based on intent
      const servers = mockGetActivatedServers(intentResult.intent);
      setActivatedServers(servers);

      // Generate mock bot response
      const botResponse: Message = {
        id: (Date.now() + 1).toString(),
        text: `I understand you're asking about "${intentResult.intent}". I've activated ${servers.length} MCP server(s) to help with your request.`,
        sender: 'bot',
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, botResponse]);
    } catch (error) {
      console.error('Error processing message:', error);
      const errorResponse: Message = {
        id: (Date.now() + 1).toString(),
        text: 'Sorry, I encountered an error processing your request.',
        sender: 'bot',
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorResponse]);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="chatbot-page">
      <div className="chat-container">
        <div className="chat-messages">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`message ${message.sender === 'user' ? 'user-message' : 'bot-message'}`}
            >
              <div className="message-content">{message.text}</div>
              <div className="message-timestamp">
                {message.timestamp.toLocaleTimeString()}
              </div>
            </div>
          ))}
          {isProcessing && (
            <div className="message bot-message">
              <div className="message-content">
                <span className="spinner">Processing...</span>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <form onSubmit={handleSubmit} className="chat-input-form">
          <input
            type="text"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder="Type your message..."
            disabled={isProcessing}
            className="chat-input"
          />
          <button
            type="submit"
            disabled={isProcessing || !inputText.trim()}
            className="submit-button"
          >
            Submit
          </button>
        </form>
      </div>

      <div className="sidebar">
        <h2>Activated MCP Servers</h2>
        {activatedServers.length === 0 ? (
          <p className="no-servers">No servers activated yet</p>
        ) : (
          <ul className="server-list">
            {activatedServers.map((server, index) => (
              <li key={index} className="server-item">
                <div className="server-name">{server.name}</div>
                <div className="server-description">{server.description}</div>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

