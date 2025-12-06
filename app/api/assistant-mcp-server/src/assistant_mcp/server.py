#!/usr/bin/env python3
"""HTTP wrapper for MCP server - makes it callable from web apps."""

import sys
from pathlib import Path
import os
import asyncio
import json
from typing import Any, Dict, Optional
from intent_classifier import IntentClassifier, Vocabulary

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from services.calendar import CalendarService
from services.alarm import AlarmService
from services.shopping import ShoppingListService
from services.media import MediaService
from services.smart_home import SmartHomeService
from services.phone import PhoneService

sys.modules['__main__'].Vocabulary = Vocabulary

# Initialize FastAPI app
app = FastAPI(title="Assistant Tools API")

# Enable CORS for web app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
calendar_service = CalendarService()
alarm_service = AlarmService()
shopping_service = ShoppingListService()
media_service = MediaService()
smart_home_service = SmartHomeService()
phone_service = PhoneService()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'best_model_with_oos.pt')
VOCAB_PATH = os.path.join(BASE_DIR, 'vocab.pkl')

print(f"BASE_DIR: {BASE_DIR}")
print(f"MODEL_PATH: {MODEL_PATH}")
print(f"VOCAB_PATH: {VOCAB_PATH}")
print(f"Model exists: {os.path.exists(MODEL_PATH)}")
print(f"Vocab exists: {os.path.exists(VOCAB_PATH)}")

# Then load
intent_classifier = IntentClassifier(model_path=None)
intent_classifier.load_vocab(VOCAB_PATH)
intent_classifier.load_model(MODEL_PATH)
print(f"✓ Intent classifier loaded from {BASE_DIR}")

class ToolCall(BaseModel):
    """Request model for tool execution."""
    name: str
    arguments: Dict[str, Any]


class ToolResponse(BaseModel):
    """Response model for tool execution."""
    result: str
    error: bool = False


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "Assistant Tools API is running"}


@app.get("/tools")
async def list_tools(query: Optional[str] = None):
    """List all available tools with complete schemas."""
    
    all_tools = [
        {
            "name": "calendar",
            "description": "View calendar events for a specific date or date range",
            "input_schema": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Date to view (YYYY-MM-DD format)"
                    },
                    "end_date": {
                        "type": "string",
                        "description": "Optional end date for range (YYYY-MM-DD format)"
                    }
                },
                "required": ["date"]
            }
        },
        {
            "name": "calendar_update",
            "description": "Add, modify, or delete calendar events",
            "input_schema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["add", "modify", "delete"],
                        "description": "Action to perform"
                    },
                    "title": {
                        "type": "string",
                        "description": "Event title"
                    },
                    "date": {
                        "type": "string",
                        "description": "Event date (YYYY-MM-DD)"
                    },
                    "time": {
                        "type": "string",
                        "description": "Event time (HH:MM)"
                    },
                    "duration": {
                        "type": "string",
                        "description": "Event duration (e.g., '1 hour', '30 minutes')"
                    },
                    "event_id": {
                        "type": "string",
                        "description": "Event ID for modify/delete actions"
                    }
                },
                "required": ["action"]
            }
        },
        {
            "name": "alarm",
            "description": "Set, view, or delete alarms",
            "input_schema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["set", "list", "delete", "delete_all"],
                        "description": "Action to perform"
                    },
                    "time": {
                        "type": "string",
                        "description": "Alarm time (HH:MM)"
                    },
                    "label": {
                        "type": "string",
                        "description": "Optional alarm label"
                    },
                    "repeat": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
                        },
                        "description": "Days to repeat alarm"
                    },
                    "alarm_id": {
                        "type": "string",
                        "description": "Alarm ID for delete action"
                    }
                },
                "required": ["action"]
            }
        },
        {
            "name": "change_speed",
            "description": "Change playback speed or speech rate",
            "input_schema": {
                "type": "object",
                "properties": {
                    "speed": {
                        "type": "number",
                        "description": "Speed multiplier (0.5 = half speed, 2.0 = double speed)",
                        "minimum": 0.5,
                        "maximum": 2.0
                    },
                    "context": {
                        "type": "string",
                        "enum": ["music", "speech", "video", "podcast"],
                        "description": "What to adjust speed for"
                    }
                },
                "required": ["speed"]
            }
        },
        {
            "name": "change_volume",
            "description": "Adjust volume level",
            "input_schema": {
                "type": "object",
                "properties": {
                    "level": {
                        "type": "number",
                        "description": "Volume level (0-100) or relative change"
                    },
                    "relative": {
                        "type": "boolean",
                        "description": "If true, level is relative change (e.g., +10, -20)",
                        "default": False
                    },
                    "device": {
                        "type": "string",
                        "description": "Specific device (e.g., 'speaker', 'headphones', 'system')"
                    }
                },
                "required": ["level"]
            }
        },
        {
            "name": "shopping_list",
            "description": "View all items on the shopping list",
            "input_schema": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Optional category filter (e.g., produce, dairy, meat)"
                    },
                    "checked": {
                        "type": "boolean",
                        "description": "Filter by checked status"
                    }
                }
            }
        },
        {
            "name": "shopping_list_update",
            "description": "Add, remove, or check off items on shopping list",
            "input_schema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["add", "remove", "check", "uncheck", "clear_checked"],
                        "description": "Action to perform"
                    },
                    "item": {
                        "type": "string",
                        "description": "Item name"
                    },
                    "quantity": {
                        "type": "string",
                        "description": "Optional quantity (e.g., '2 pounds', '1 gallon')"
                    },
                    "category": {
                        "type": "string",
                        "description": "Optional category (e.g., produce, dairy)"
                    }
                },
                "required": ["action", "item"]
            }
        },
        {
            "name": "timer",
            "description": "Set, view, or cancel timers",
            "input_schema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["set", "list", "cancel", "pause", "resume", "cancel_all"],
                        "description": "Action to perform"
                    },
                    "duration": {
                        "type": "string",
                        "description": "Timer duration (e.g., '5 minutes', '1 hour 30 minutes', '45 seconds')"
                    },
                    "label": {
                        "type": "string",
                        "description": "Optional timer label"
                    },
                    "timer_id": {
                        "type": "string",
                        "description": "Timer ID for cancel/pause/resume actions"
                    }
                },
                "required": ["action"]
            }
        },
        {
            "name": "find_phone",
            "description": "Locate the user's phone by making it ring or showing location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "method": {
                        "type": "string",
                        "enum": ["ring", "location", "both"],
                        "description": "How to find the phone"
                    }
                },
                "required": ["method"]
            }
        },
        {
            "name": "play_music",
            "description": "Play music by artist, song, album, genre, or playlist",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to play (song name, artist, genre, etc.)"
                    },
                    "type": {
                        "type": "string",
                        "enum": ["song", "artist", "album", "genre", "playlist", "podcast"],
                        "description": "Type of music query"
                    },
                    "shuffle": {
                        "type": "boolean",
                        "description": "Whether to shuffle",
                        "default": False
                    },
                    "source": {
                        "type": "string",
                        "description": "Music source (e.g., 'spotify', 'apple_music', 'youtube')"
                    }
                },
                "required": ["query"]
            }
        },
        {
            "name": "smart_home",
            "description": "Control smart home devices (lights, thermostat, locks, etc.)",
            "input_schema": {
                "type": "object",
                "properties": {
                    "device_type": {
                        "type": "string",
                        "enum": ["light", "thermostat", "lock", "blinds", "outlet", "fan", "camera"],
                        "description": "Type of device to control"
                    },
                    "device_name": {
                        "type": "string",
                        "description": "Specific device name or 'all' for all devices of that type"
                    },
                    "action": {
                        "type": "string",
                        "description": "Action to perform (e.g., 'on', 'off', 'dim', 'set temperature', 'lock', 'unlock')"
                    },
                    "value": {
                        "type": "string",
                        "description": "Value for the action (e.g., brightness %, temperature, color)"
                    },
                    "room": {
                        "type": "string",
                        "description": "Room name to control all devices in that room"
                    }
                },
                "required": ["device_type", "action"]
            }
        }
    ]
    
    # If no query provided, return all tools
    if not query:
        return {"tools": all_tools}
    
    # Use intent classifier to filter tools
    prediction = intent_classifier.predict_intent(query)
    relevant_tool_names = intent_classifier.get_relevant_tools(query)
    
    if prediction['is_oos'] or not relevant_tool_names:
        filtered_tools = []
    else:
        filtered_tools = [tool for tool in all_tools if tool['name'] in relevant_tool_names]
    
    return {
        "tools": filtered_tools,
        "intent": prediction['intent'],
        "confidence": prediction['confidence']
    }


@app.post("/execute", response_model=ToolResponse)
async def execute_tool(tool_call: ToolCall):
    """Execute a tool with given arguments."""
    try:
        name = tool_call.name
        args = tool_call.arguments
        
        # Route to appropriate service
        if name == "calendar":
            result = await calendar_service.get_events(
                date=args["date"],
                end_date=args.get("end_date")
            )
            
        elif name == "calendar_update":
            action = args["action"]
            if action == "add":
                result = await calendar_service.add_event(
                    title=args["title"],
                    date=args["date"],
                    time=args.get("time"),
                    duration=args.get("duration")
                )
            elif action == "modify":
                result = await calendar_service.modify_event(
                    event_id=args["event_id"],
                    title=args.get("title"),
                    date=args.get("date"),
                    time=args.get("time"),
                    duration=args.get("duration")
                )
            elif action == "delete":
                result = await calendar_service.delete_event(
                    event_id=args["event_id"]
                )
                
        elif name == "alarm":
            action = args["action"]
            if action == "set":
                result = await alarm_service.set_alarm(
                    time=args["time"],
                    label=args.get("label"),
                    repeat=args.get("repeat", [])
                )
            elif action == "list":
                result = await alarm_service.list_alarms()
            elif action == "delete":
                result = await alarm_service.delete_alarm(
                    alarm_id=args["alarm_id"]
                )
            elif action == "delete_all":
                result = await alarm_service.delete_all_alarms()
                
        elif name == "change_speed":
            result = await media_service.change_speed(
                speed=args["speed"],
                context=args.get("context", "music")
            )
            
        elif name == "change_volume":
            result = await media_service.change_volume(
                level=args["level"],
                relative=args.get("relative", False),
                device=args.get("device")
            )
            
        elif name == "shopping_list":
            result = await shopping_service.get_list(
                category=args.get("category"),
                checked=args.get("checked")
            )
            
        elif name == "shopping_list_update":
            action = args["action"]
            if action == "add":
                result = await shopping_service.add_item(
                    item=args["item"],
                    quantity=args.get("quantity"),
                    category=args.get("category")
                )
            elif action == "remove":
                result = await shopping_service.remove_item(
                    item=args["item"]
                )
            elif action == "check":
                result = await shopping_service.check_item(
                    item=args["item"]
                )
            elif action == "uncheck":
                result = await shopping_service.uncheck_item(
                    item=args["item"]
                )
            elif action == "clear_checked":
                result = await shopping_service.clear_checked()
                
        elif name == "timer":
            action = args["action"]
            if action == "set":
                result = await media_service.set_timer(
                    duration=args["duration"],
                    label=args.get("label")
                )
            elif action == "list":
                result = await media_service.list_timers()
            elif action == "cancel":
                result = await media_service.cancel_timer(
                    timer_id=args["timer_id"]
                )
            elif action == "pause":
                result = await media_service.pause_timer(
                    timer_id=args["timer_id"]
                )
            elif action == "resume":
                result = await media_service.resume_timer(
                    timer_id=args["timer_id"]
                )
            elif action == "cancel_all":
                result = await media_service.cancel_all_timers()
                
        elif name == "find_phone":
            result = await phone_service.find_phone(
                method=args["method"]
            )
            
        elif name == "play_music":
            result = await media_service.play_music(
                query=args["query"],
                type_=args.get("type", "song"),
                shuffle=args.get("shuffle", False),
                source=args.get("source")
            )
            
        elif name == "smart_home":
            result = await smart_home_service.control_device(
                device_type=args["device_type"],
                device_name=args.get("device_name"),
                action=args["action"],
                value=args.get("value"),
                room=args.get("room")
            )
            
        else:
            raise ValueError(f"Unknown tool: {name}")
        
        return ToolResponse(result=result, error=False)
        
    except Exception as e:
        return ToolResponse(result=str(e), error=True)


if __name__ == "__main__":
    import uvicorn
    print("Starting Assistant Tools HTTP Server on http://localhost:8080")
    uvicorn.run(app, host="0.0.0.0", port=8080)