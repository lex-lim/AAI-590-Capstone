import { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

export default function LandingPage() {
  const [isAuthenticating, setIsAuthenticating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const navigate = useNavigate();

  useEffect(() => {
    // Cleanup stream on unmount
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  const captureFrames = async (video: HTMLVideoElement, count: number = 5, interval: number = 200): Promise<string[]> => {
    const frames: string[] = [];
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');

    if (!ctx) {
      return frames;
    }

    // Wait for video to be ready
    if (video.readyState < 2) {
      await new Promise((resolve) => {
        video.addEventListener('loadeddata', resolve, { once: true });
      });
    }

    for (let i = 0; i < count; i++) {
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const base64 = canvas.toDataURL('image/jpeg', 0.8);
      frames.push(base64);
      
      // Wait before capturing next frame (except for the last one)
      if (i < count - 1) {
        await new Promise((resolve) => setTimeout(resolve, interval));
      }
    }

    return frames;
  };

  const authenticateWithBackend = async (frames: string[]): Promise<{ success: boolean; user?: string }> => {
    const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
    
    try {
      const response = await fetch(`${API_URL}/api/auth/face`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ frames }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Authentication failed' }));
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      return {
        success: data.success,
        user: data.user,
      };
    } catch (err) {
      console.error('API error:', err);
      throw err;
    }
  };

  const handleLogin = async () => {
    setIsAuthenticating(true);
    setError(null);

    try {
      // Request webcam access
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }

      // Wait for webcam to start and stabilize
      await new Promise((resolve) => setTimeout(resolve, 1000));

      // Capture frames from webcam
      if (!videoRef.current) {
        throw new Error('Video element not available');
      }

      const frames = await captureFrames(videoRef.current, 5, 300);
      
      if (frames.length === 0) {
        throw new Error('Could not capture frames from webcam');
      }

      // Authenticate with backend
      const result = await authenticateWithBackend(frames);

      // Stop webcam stream
      stream.getTracks().forEach((track) => track.stop());
      streamRef.current = null;

      if (result.success && result.user) {
        // Store authenticated user name
        localStorage.setItem('authenticatedUser', result.user);
        // Navigate to chatbot page
        navigate('/chat');
      } else {
        setError('Authentication failed. Please try again.');
        setIsAuthenticating(false);
      }
    } catch (err) {
      console.error('Error during authentication:', err);
      const errorMessage = err instanceof Error ? err.message : 'Authentication failed';
      setError(errorMessage);
      setIsAuthenticating(false);
      
      // Clean up stream if it was partially created
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      }
    }
  };

  return (
    <div className="landing-page">
      <div className="landing-content">
        <h1>Welcome</h1>
        <p>Please log in to continue</p>
        
        {error && <div className="error-message">{error}</div>}
        
        {isAuthenticating ? (
          <div className="auth-container">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              className="webcam-video"
            />
            <div className="auth-status">Authenticating...</div>
          </div>
        ) : (
          <button onClick={handleLogin} className="login-button">
            Login
          </button>
        )}
      </div>
    </div>
  );
}

