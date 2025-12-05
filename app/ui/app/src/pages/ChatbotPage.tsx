import { useState, useRef, useEffect } from 'react';
import { mockClassifyIntent, mockGetActivatedServers, type MCPServer } from '../services/mockApi';
import { client } from './client';

interface Message {
  id: string;
  text: string;
  sender: 'user' | 'assistant';
  timestamp: Date;
}

export default function ChatbotPage() {
  // Get authenticated user name from localStorage
  const authenticatedUser = localStorage.getItem('authenticatedUser') || 'there';

  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      text: `Hello ${authenticatedUser}! How can I help you today?`,
      sender: 'assistant',
      timestamp: new Date(),
    },
  ]);
  const [inputText, setInputText] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [activatedServers, setActivatedServers] = useState<MCPServer[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const PYTHON_API_URL = 'http://localhost:8080';

  useEffect(() => {
    // Auto-scroll to bottom when new messages are added
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Fetch tools from Python server
  const getTools = async () => {
    try {
      const response = await fetch(`${PYTHON_API_URL}/tools`);
      const data = await response.json();
      return data.tools;
    } catch (error) {
      console.error('Error fetching tools:', error);
      return [];
    }
  };

  // Execute tool via Python server
  const executeTool = async (name: string, args: any) => {
    const response = await fetch(`${PYTHON_API_URL}/execute`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, arguments: args })
    });
    
    const data = await response.json();
    
    if (data.error) {
      throw new Error(data.result);
    }
    
    return data.result;
  };

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

    const history = messages.map(m => ({
      role: m.sender,
      content: m.text
    }));

    history.push({
      role: userMessage.sender,
      content: userMessage.text
    });

    try {
      console.log('Fetching tools from Python server...');
      const tools = await getTools();
      console.log(`Loaded ${tools.length} tools`);

      // Update activated servers display
      if (tools.length > 0) {
        setActivatedServers([{
          name: 'Assistant Tools',
          description: `${tools.length} tools available: ${tools.map((t: any) => t.name).join(', ')}`
        }]);
      }

      console.log('Calling Claude with tools...');
      let response = await client.beta.messages.create({
        max_tokens: 2000,
        messages: history,
        model: 'claude-sonnet-4-5-20250929',
        tools: tools, // Pass tools from Python server
      });

      let conversationHistory = [...history];
      let iterations = 0;
      const MAX_ITERATIONS = 5;

      // Handle tool calls
      while (response.content.some((block: any) => block.type === 'tool_use') && iterations < MAX_ITERATIONS) {
        iterations++;
        console.log(`Tool call iteration ${iterations}`);

        const toolUses = response.content.filter((block: any) => block.type === 'tool_use');
        console.log(`Claude wants to use ${toolUses.length} tool(s):`, toolUses.map((t: any) => t.name));

        // Execute tools via Python server
        const toolResults = await Promise.all(
          toolUses.map(async (toolUse: any) => {
            console.log(`Executing: ${toolUse.name}`, toolUse.input);

            try {
              const result = await executeTool(toolUse.name, toolUse.input);
              console.log(`Result from ${toolUse.name}:`, result);

              return {
                type: 'tool_result',
                tool_use_id: toolUse.id,
                content: result
              };
            } catch (error: any) {
              console.error(`Error executing ${toolUse.name}:`, error);
              return {
                type: 'tool_result',
                tool_use_id: toolUse.id,
                content: `Error: ${error.message}`,
                is_error: true
              };
            }
          })
        );

        // Add assistant response and tool results to conversation
        conversationHistory.push(
          { role: 'assistant', content: response.content },
          { role: 'user', content: toolResults }
        );

        console.log('Sending tool results back to Claude...');
        response = await client.beta.messages.create({
          max_tokens: 2000,
          messages: conversationHistory,
          model: 'claude-sonnet-4-5-20250929',
          tools: tools,
        });
      }

      // Extract final text response
      const textBlocks = response.content
        .filter((block: any) => block.type === "text")
        .map((block: any) => block.text);

      const allText = textBlocks.join("\n");

      // Generate bot response
      const botResponse: Message = {
        id: (Date.now() + 1).toString(),
        text: allText,
        sender: 'assistant',
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, botResponse]);
    } catch (error) {
      console.error('Error processing message:', error);
      const errorResponse: Message = {
        id: (Date.now() + 1).toString(),
        text: `Sorry, I encountered an error: ${error instanceof Error ? error.message : 'Unknown error'}`,
        sender: 'assistant',
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