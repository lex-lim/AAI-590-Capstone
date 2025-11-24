from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models import FaceAuthRequest, FaceAuthResponse
from face_classifier import load_model_and_face_detector, process_frames


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and face detector on startup."""
    print("Starting up... Loading face classification model...")
    model, face_cascade, class_names = load_model_and_face_detector()
    yield  


app = FastAPI(
    title="Face Authentication API",
    description="API for face-based user authentication",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Vite default ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    model, face_cascade, _ = load_model_and_face_detector()
    if model is None or face_cascade is None:
        return {"status": "unhealthy", "error": "Model or face detector not loaded"}
    return {"status": "healthy"}


@app.post("/api/auth/face", response_model=FaceAuthResponse)
async def authenticate_face(request: FaceAuthRequest):
    """
    Authenticate user based on face images.
    
    Accepts an array of base64-encoded images, processes them to detect
    and classify faces, and returns the identified user.
    """
    predicted_user, confidence, details = process_frames(request.frames)
    
    if predicted_user is None:
        return FaceAuthResponse(
            success=False,
            error=details.get("error", "Could not identify user"),
            details=details
        )
    
    return FaceAuthResponse(
        success=True,
        user=predicted_user,
        confidence=confidence,
        details=details
    )

# TODO: Add intent classification endpoint
# TODO: Add a endpoint to get the activated servers based on the intent