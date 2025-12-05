"""Calendar service for managing events."""

import uuid
from datetime import datetime, timedelta
from typing import Optional


class CalendarService:
    """Service for calendar operations."""
    
    def __init__(self):
        # In-memory storage (replace with actual calendar API)
        self.events = {}
    
    async def get_events(self, date: str, end_date: Optional[str] = None) -> str:
        """Get events for a date or date range."""
        # TODO: Integrate with Google Calendar, Outlook, etc.
        try:
            start = datetime.strptime(date, "%Y-%m-%d")
            if end_date:
                end = datetime.strptime(end_date, "%Y-%m-%d")
            else:
                end = start
            
            # Filter events in date range
            matching_events = [
                event for event in self.events.values()
                if start <= datetime.strptime(event["date"], "%Y-%m-%d") <= end
            ]
            
            if not matching_events:
                return f"No events found for {date}" + (f" to {end_date}" if end_date else "")
            
            result = f"Events for {date}" + (f" to {end_date}" if end_date else "") + ":\n"
            for event in sorted(matching_events, key=lambda e: (e["date"], e.get("time", ""))):
                time_str = f" at {event['time']}" if event.get("time") else ""
                result += f"- {event['title']}{time_str}\n"
            
            return result.strip()
            
        except ValueError as e:
            return f"Invalid date format. Use YYYY-MM-DD. Error: {str(e)}"
    
    async def add_event(
        self, 
        title: str, 
        date: str, 
        time: Optional[str] = None,
        duration: Optional[str] = None
    ) -> str:
        """Add a new calendar event."""
        try:
            # Validate date
            datetime.strptime(date, "%Y-%m-%d")
            if time:
                datetime.strptime(time, "%H:%M")
            
            event_id = str(uuid.uuid4())
            self.events[event_id] = {
                "id": event_id,
                "title": title,
                "date": date,
                "time": time,
                "duration": duration
            }
            
            time_str = f" at {time}" if time else ""
            duration_str = f" ({duration})" if duration else ""
            return f"Added event '{title}' on {date}{time_str}{duration_str}"
            
        except ValueError as e:
            return f"Invalid date/time format. Error: {str(e)}"
    
    async def modify_event(
        self,
        event_id: str,
        title: Optional[str] = None,
        date: Optional[str] = None,
        time: Optional[str] = None,
        duration: Optional[str] = None
    ) -> str:
        """Modify an existing event."""
        if event_id not in self.events:
            return f"Event {event_id} not found"
        
        event = self.events[event_id]
        if title:
            event["title"] = title
        if date:
            event["date"] = date
        if time:
            event["time"] = time
        if duration:
            event["duration"] = duration
        
        return f"Updated event: {event['title']}"
    
    async def delete_event(self, event_id: str) -> str:
        """Delete an event."""
        if event_id not in self.events:
            return f"Event {event_id} not found"
        
        event = self.events.pop(event_id)
        return f"Deleted event: {event['title']}"
