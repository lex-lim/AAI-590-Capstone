"""Alarm service for managing alarms."""

import uuid
from typing import List, Optional


class AlarmService:
    """Service for alarm operations."""
    
    def __init__(self):
        # In-memory storage (replace with actual alarm system)
        self.alarms = {}
    
    async def set_alarm(
        self, 
        time: str, 
        label: Optional[str] = None,
        repeat: Optional[List[str]] = None
    ) -> str:
        """Set a new alarm."""
        alarm_id = str(uuid.uuid4())
        self.alarms[alarm_id] = {
            "id": alarm_id,
            "time": time,
            "label": label,
            "repeat": repeat or [],
            "enabled": True
        }
        
        label_str = f" ({label})" if label else ""
        repeat_str = f" repeating on {', '.join(repeat)}" if repeat else ""
        return f"Alarm set for {time}{label_str}{repeat_str}"
    
    async def list_alarms(self) -> str:
        """List all alarms."""
        if not self.alarms:
            return "No alarms set"
        
        result = "Active alarms:\n"
        for alarm in sorted(self.alarms.values(), key=lambda a: a["time"]):
            label_str = f" - {alarm['label']}" if alarm["label"] else ""
            repeat_str = f" (repeats: {', '.join(alarm['repeat'])})" if alarm["repeat"] else ""
            status = "enabled" if alarm["enabled"] else "disabled"
            result += f"- {alarm['time']}{label_str}{repeat_str} [{status}]\n"
        
        return result.strip()
    
    async def delete_alarm(self, alarm_id: str) -> str:
        """Delete a specific alarm."""
        if alarm_id not in self.alarms:
            return f"Alarm {alarm_id} not found"
        
        alarm = self.alarms.pop(alarm_id)
        label_str = f" ({alarm['label']})" if alarm["label"] else ""
        return f"Deleted alarm for {alarm['time']}{label_str}"
    
    async def delete_all_alarms(self) -> str:
        """Delete all alarms."""
        count = len(self.alarms)
        self.alarms.clear()
        return f"Deleted {count} alarm(s)"
