"""Media service for music, timers, and playback control."""

import uuid
from datetime import datetime, timedelta
from typing import Optional
import re


class MediaService:
    """Service for media operations."""
    
    def __init__(self):
        self.timers = {}
        self.current_volume = 50
        self.current_speed = 1.0
    
    async def change_speed(self, speed: float, context: str = "music") -> str:
        """Change playback speed."""
        self.current_speed = speed
        return f"{context.capitalize()} speed set to {speed}x"
    
    async def change_volume(
        self, 
        level: float, 
        relative: bool = False,
        device: Optional[str] = None
    ) -> str:
        """Change volume level."""
        if relative:
            self.current_volume = max(0, min(100, self.current_volume + level))
            action = "increased" if level > 0 else "decreased"
            device_str = f" on {device}" if device else ""
            return f"Volume {action} by {abs(level)}%{device_str} (now at {self.current_volume}%)"
        else:
            self.current_volume = max(0, min(100, level))
            device_str = f" on {device}" if device else ""
            return f"Volume set to {self.current_volume}%{device_str}"
    
    async def set_timer(self, duration: str, label: Optional[str] = None) -> str:
        """Set a timer."""
        # Parse duration (e.g., "5 minutes", "1 hour 30 minutes", "45 seconds")
        seconds = self._parse_duration(duration)
        if seconds is None:
            return f"Invalid duration format: {duration}"
        
        timer_id = str(uuid.uuid4())
        end_time = datetime.now() + timedelta(seconds=seconds)
        
        self.timers[timer_id] = {
            "id": timer_id,
            "duration": duration,
            "seconds": seconds,
            "label": label,
            "end_time": end_time,
            "paused": False,
            "remaining": seconds
        }
        
        label_str = f" ({label})" if label else ""
        return f"Timer set for {duration}{label_str}"
    
    def _parse_duration(self, duration: str) -> Optional[int]:
        """Parse duration string to seconds."""
        duration = duration.lower()
        total_seconds = 0
        
        # Match hours
        hours = re.search(r'(\d+)\s*(?:hour|hr|h)', duration)
        if hours:
            total_seconds += int(hours.group(1)) * 3600
        
        # Match minutes
        minutes = re.search(r'(\d+)\s*(?:minute|min|m)', duration)
        if minutes:
            total_seconds += int(minutes.group(1)) * 60
        
        # Match seconds
        seconds = re.search(r'(\d+)\s*(?:second|sec|s)', duration)
        if seconds:
            total_seconds += int(seconds.group(1))
        
        # If only a number, assume minutes
        if total_seconds == 0:
            number = re.search(r'^(\d+)$', duration)
            if number:
                total_seconds = int(number.group(1)) * 60
        
        return total_seconds if total_seconds > 0 else None
    
    async def list_timers(self) -> str:
        """List all active timers."""
        if not self.timers:
            return "No active timers"
        
        result = "Active timers:\n"
        now = datetime.now()
        
        for timer in sorted(self.timers.values(), key=lambda t: t["end_time"]):
            if timer["paused"]:
                remaining = timer["remaining"]
                status = "paused"
            else:
                remaining = max(0, (timer["end_time"] - now).total_seconds())
                status = "running"
            
            label_str = f" - {timer['label']}" if timer["label"] else ""
            mins, secs = divmod(int(remaining), 60)
            hours, mins = divmod(mins, 60)
            
            if hours > 0:
                time_str = f"{hours}h {mins}m"
            elif mins > 0:
                time_str = f"{mins}m {secs}s"
            else:
                time_str = f"{secs}s"
            
            result += f"- {time_str} remaining{label_str} [{status}]\n"
        
        return result.strip()
    
    async def cancel_timer(self, timer_id: str) -> str:
        """Cancel a specific timer."""
        if timer_id not in self.timers:
            return f"Timer {timer_id} not found"
        
        timer = self.timers.pop(timer_id)
        label_str = f" ({timer['label']})" if timer["label"] else ""
        return f"Cancelled timer{label_str}"
    
    async def pause_timer(self, timer_id: str) -> str:
        """Pause a timer."""
        if timer_id not in self.timers:
            return f"Timer {timer_id} not found"
        
        timer = self.timers[timer_id]
        if timer["paused"]:
            return "Timer is already paused"
        
        now = datetime.now()
        timer["remaining"] = max(0, (timer["end_time"] - now).total_seconds())
        timer["paused"] = True
        
        label_str = f" ({timer['label']})" if timer["label"] else ""
        return f"Paused timer{label_str}"
    
    async def resume_timer(self, timer_id: str) -> str:
        """Resume a paused timer."""
        if timer_id not in self.timers:
            return f"Timer {timer_id} not found"
        
        timer = self.timers[timer_id]
        if not timer["paused"]:
            return "Timer is not paused"
        
        timer["end_time"] = datetime.now() + timedelta(seconds=timer["remaining"])
        timer["paused"] = False
        
        label_str = f" ({timer['label']})" if timer["label"] else ""
        return f"Resumed timer{label_str}"
    
    async def cancel_all_timers(self) -> str:
        """Cancel all timers."""
        count = len(self.timers)
        self.timers.clear()
        return f"Cancelled {count} timer(s)"
    
    async def play_music(
        self,
        query: str,
        type_: str = "song",
        shuffle: bool = False,
        source: Optional[str] = None
    ) -> str:
        """Play music."""
        # TODO: Integrate with Spotify, Apple Music, etc.
        shuffle_str = " (shuffled)" if shuffle else ""
        source_str = f" on {source}" if source else ""
        
        return f"Now playing {type_}: '{query}'{shuffle_str}{source_str}"
