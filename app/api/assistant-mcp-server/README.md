# Assistant MCP Server

An MCP (Model Context Protocol) server providing personal assistant tools for calendar, alarms, shopping lists, media control, smart home, and more.

## Installation

```bash
pip install mcp
pip install -e .
```

## Usage

### Start the server

```bash
# Option 1: Direct run
python -m assistant_mcp.server

# Option 2: After install
assistant-mcp-server
```

### With Claude Desktop/API

Add to your MCP settings:

```json
{
  "mcpServers": {
    "assistant": {
      "command": "python",
      "args": ["-m", "assistant_mcp.server"]
    }
  }
}
```

### With Node.js Backend

```javascript
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';

const transport = new StdioClientTransport({
  command: 'python',
  args: ['-m', 'assistant_mcp.server']
});

const mcpClient = new Client({
  name: 'web-backend',
  version: '1.0.0'
}, {
  capabilities: {}
});

await mcpClient.connect(transport);

// Get available tools
const { tools } = await mcpClient.listTools();
```

## Available Tools

- **calendar** - View calendar events
- **calendar_update** - Add/modify/delete events
- **alarm** - Set/view/delete alarms
- **change_speed** - Adjust playback speed
- **change_volume** - Control volume
- **shopping_list** - View shopping list
- **shopping_list_update** - Manage shopping items
- **timer** - Set/manage timers
- **find_phone** - Locate phone
- **play_music** - Play music
- **smart_home** - Control smart devices

## Project Structure

```
assistant-mcp-server/
├── pyproject.toml
├── README.md
├── src/
│   └── assistant_mcp/
│       ├── __init__.py
│       ├── server.py
│       └── services/
│           ├── __init__.py
│           ├── calendar.py
│           ├── alarm.py
│           ├── shopping.py
│           ├── media.py
│           ├── smart_home.py
│           └── phone.py
```

## Development

The service implementations in `services/` are placeholders. Replace them with actual integrations:

- Calendar: Google Calendar API, Outlook, etc.
- Smart Home: Home Assistant, SmartThings, etc.
- Music: Spotify, Apple Music, etc.

## License

MIT
