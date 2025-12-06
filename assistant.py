#!/usr/bin/env python3
"""
Wake Word Detection for AI Assistant
Opens browser in new window when wake word is detected
"""

import sys
import os
import webbrowser
import time

# Wake word detection imports
try:
    import pvporcupine
    from pvrecorder import PvRecorder
except ImportError:
    print("[ERROR] Wake word libraries not installed")
    print("Install with: pip install pvporcupine pvrecorder")
    sys.exit(1)


class WakeWordBrowserLauncher:
    def __init__(self, access_key=None, keywords=["computer"], url="http://localhost:5173"):
        """
        Initialize the wake word browser launcher.
        
        Args:
            access_key: Picovoice access key
            keywords: List of wake words to detect
            url: URL to open in browser when wake word is detected
        """
        self.access_key = access_key
        self.keywords = keywords
        self.url = url
        self.porcupine = None
        self.recorder = None
        self.is_running = False
        self.paused = False
        
    def initialize(self):
        """Initialize Porcupine and audio recorder."""
        try:
            # Initialize Porcupine
            self.porcupine = pvporcupine.create(
                access_key=self.access_key,
                keywords=self.keywords
            )
            
            # Initialize audio recorder
            self.recorder = PvRecorder(
                device_index=-1,  # Use default audio device
                frame_length=self.porcupine.frame_length
            )
            
            print(f"[OK] Wake word detector initialized")
            print(f"     Wake word(s): {', '.join(self.keywords)}")
            print(f"     Target URL: {self.url}")
            print(f"     Audio device: {self.recorder.selected_device}")
            return True
            
        except Exception as e:
            print(f"[ERROR] Initialization failed: {e}")
            return False
    
    def open_browser(self):
        """Open the URL in a new browser window."""
        try:
            print(f"\n[BROWSER] Opening: {self.url}")
            # Open in new window
            webbrowser.open_new(self.url)
            print(f"[OK] Browser opened successfully!")
            
        except Exception as e:
            print(f"[ERROR] Failed to open browser: {e}")
    
    def start(self):
        """Start listening for wake word."""
        if not self.initialize():
            return
        
        self.is_running = True
        self.recorder.start()
        
        print("\n" + "="*60)
        print("LISTENING FOR WAKE WORD...")
        print("="*60)
        print(f"Say '{self.keywords[0]}' to open browser in new window")
        print(f"Say '{self.keywords[0]}' again while paused to resume listening")
        print("Press Ctrl+C to stop\n")
        
        try:
            while self.is_running:
                # Read audio frame
                pcm = self.recorder.read()
                
                # Process audio frame
                keyword_index = self.porcupine.process(pcm)
                
                # Wake word detected
                if keyword_index >= 0:
                    detected_keyword = self.keywords[keyword_index]
                    print(f"\n[DETECTED] Wake word '{detected_keyword}' detected!")
                    
                    if not self.paused:
                        # Open browser and pause
                        self.open_browser()
                        self.paused = True
                        print("\n" + "="*60)
                        print("[PAUSED] Wake word detection stopped")
                        print("="*60)
                        print(f"Close your browser tab, then say '{detected_keyword}' to resume")
                        print("Or press Ctrl+C to exit\n")
                    else:
                        # Resume listening
                        self.paused = False
                        print("\n" + "="*60)
                        print("[RESUMED] Listening for wake word...")
                        print("="*60)
                        print(f"Say '{detected_keyword}' to open browser again\n")
                    
        except KeyboardInterrupt:
            print("\n\n[STOPPING] Shutting down wake word detector...")
        except Exception as e:
            print(f"\n[ERROR] {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop()
    
    def stop(self):
        """Stop the detector and clean up resources."""
        self.is_running = False
        
        if self.recorder:
            try:
                self.recorder.stop()
            except:
                pass
            self.recorder.delete()
        
        if self.porcupine:
            self.porcupine.delete()
        
        print("[OK] Cleanup complete")


def main():
    # Configuration
    URL = "http://localhost:5173"
    KEYWORDS = ["computer"]
    ACCESS_KEY = 'ee5XhjyCvgWH8dhY4yFfIEd5EGo9yLjJYHoMFYni2HmrkwJZBV8eNw=='
    
    # Check for access key in environment variable
    if not ACCESS_KEY:
        ACCESS_KEY = os.environ.get('PICOVOICE_ACCESS_KEY')
    
    # Create and start launcher
    launcher = WakeWordBrowserLauncher(
        access_key=ACCESS_KEY,
        keywords=KEYWORDS,
        url=URL
    )
    
    launcher.start()


if __name__ == "__main__":
    main()