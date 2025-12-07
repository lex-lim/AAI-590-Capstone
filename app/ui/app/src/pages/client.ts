import Anthropic from '@anthropic-ai/sdk'
import {apikey} from '../../../../key.ts'


export const client = new Anthropic({
  apiKey: apikey,
  dangerouslyAllowBrowser: true
})

const PYTHON_API_URL = 'http://localhost:8080';

// Get tools from Python server
async function getTools() {
  const response = await fetch(`${PYTHON_API_URL}/tools`);
  const data = await response.json();
  return data.tools;
}

// Execute tool via Python server
async function executeTool(name, args) {
  const response = await fetch(`${PYTHON_API_URL}/execute`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, args })
  });
  
  const data = await response.json();
  
  if (data.error) {
    throw new Error(data.result);
  }
  
  return data.result;
}

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const { messages } = req.body;
    
    console.log('Fetching tools from Python server...');
    const tools = await getTools();
    console.log(`Loaded ${tools.length} tools`);

    console.log('Calling Claude with tools...');
    let response = await client.messages.create({
      model: 'claude-sonnet-4-5-20250929',
      max_tokens: 2000,
      messages: messages,
      tools: tools
    });

    let conversationMessages = [...messages];
    let iterations = 0;
    const MAX_ITERATIONS = 5; // Prevent infinite loops
    
    // Handle tool calls
    while (response.content.some(block => block.type === 'tool_use') && iterations < MAX_ITERATIONS) {
      iterations++;
      console.log(`Tool call iteration ${iterations}`);
      
      const toolUses = response.content.filter(block => block.type === 'tool_use');
      console.log(`Claude wants to use ${toolUses.length} tool(s):`, toolUses.map(t => t.name));
      
      // Execute tools via Python HTTP API
      const toolResults = await Promise.all(
        toolUses.map(async (toolUse) => {
          console.log(`Executing: ${toolUse.name}`, toolUse.input);
          
          try {
            const result = await executeTool(toolUse.name, toolUse.input);
            console.log(`Result from ${toolUse.name}:`, result);
            
            return {
              type: 'tool_result',
              tool_use_id: toolUse.id,
              content: result
            };
          } catch (error) {
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
      
      conversationMessages.push(
        { role: 'assistant', content: response.content },
        { role: 'user', content: toolResults }
      );
      
      console.log('Sending tool results back to Claude...');
      response = await client.messages.create({
        model: 'claude-sonnet-4-5-20250929',
        max_tokens: 2000,
        messages: conversationMessages,
        tools: tools
      });
    }
    
    const finalText = response.content
      .filter(block => block.type === 'text')
      .map(block => block.text)
      .join('\n');
    
    console.log('Final response:', finalText);
    return res.json({ response: finalText });
    
  } catch (error) {
    console.error('Chat error:', error);
    return res.status(500).json({ 
      error: error.message,
      stack: process.env.NODE_ENV === 'development' ? error.stack : undefined
    });
  }
}