from pydantic import BaseModel
from typing import List, Optional, Dict


class FaceAuthRequest(BaseModel):
    """Request model for face authentication endpoint."""
    frames: List[str]  # Array of base64-encoded image strings


class FaceAuthResponse(BaseModel):
    """Response model for face authentication endpoint."""
    success: bool
    user: Optional[str] = None  # 'Alexis', 'Dimitri', or 'Pallav'
    confidence: float = 0.0
    details: Dict = {}
    error: Optional[str] = None

