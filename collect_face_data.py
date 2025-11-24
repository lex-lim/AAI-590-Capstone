"""
Facial Data Collection Script
Collects facial images from webcam for training a custom facial recognition model.
"""

import cv2
import os
import time
from datetime import datetime


class FaceDataCollector:
    def __init__(self, base_dir="face_data"):
        """
        Initialize the face data collector.
        
        Args:
            base_dir: Base directory to store collected face images
        """
        self.base_dir = base_dir
        self.face_cascade = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml'
)
        
        
        # Create base directory if it doesn't exist
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
    
    def create_person_directory(self, person_name):
        """Create a directory for a specific person."""
        person_dir = os.path.join(self.base_dir, person_name)
        if not os.path.exists(person_dir):
            os.makedirs(person_dir)
        return person_dir
    
    def collect_faces(self, person_name, num_images=100, capture_interval=0.3):
        """
        Collect facial images for a specific person.
        
        Args:
            person_name: Name of the person (used as folder name)
            num_images: Number of images to collect
            capture_interval: Time in seconds between captures
        """
        person_dir = self.create_person_directory(person_name)
        
        # Initialize webcam
        cap = cv2.VideoCapture(0) # Check the camera index that works for you!
        
        if not cap.isOpened():
            print("Error: Could not access webcam")
            return
        
        print(f"\n{'='*60}")
        print(f"Collecting faces for: {person_name}")
        print(f"Target images: {num_images}")
        print(f"{'='*60}")
        print("\nInstructions:")
        print("- Look at the camera and move your head slightly")
        print("- Try different angles and expressions")
        print("- Press 'q' to quit early")
        print("- Press SPACE to pause/resume")
        print("\nStarting in 3 seconds...")
        time.sleep(3)
        
        count = 0
        paused = False
        last_capture_time = 0
        
        while count < num_images:
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=8, 
                minSize=(100, 100)
            )
            
            # Draw rectangles around faces and capture
            current_time = time.time()
            
            for (x, y, w, h) in faces:
                
                # Capture face if enough time has passed and not paused
                if not paused and (current_time - last_capture_time) >= capture_interval:
                    # Extract face region with some padding
                    padding = 80
                    y1 = max(0, y - padding)
                    y2 = min(frame.shape[0], y + h + padding)
                    x1 = max(0, x - padding)
                    x2 = min(frame.shape[1], x + w + padding)
                    
                    face_img = frame[y1:y2, x1:x2]
                    
                    # Save image
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = f"{person_name}_{count:04d}_{timestamp}.jpg"
                    filepath = os.path.join(person_dir, filename)
                    cv2.imwrite(filepath, face_img)
                    
                    count += 1
                    last_capture_time = current_time
                    
                    # Print progress
                    print(f"Captured: {count}/{num_images} images", end='\r')
            # Draw rectangle
                color = (0, 255, 0) if not paused else (0, 165, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Display status on frame
            status = "PAUSED" if paused else f"Capturing: {count}/{num_images}"
            cv2.putText(frame, status, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show warning if no face detected
            if len(faces) == 0:
                cv2.putText(frame, "No face detected!", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display frame
            cv2.imshow(f'Face Collection - {person_name}', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n\nCollection stopped by user")
                break
            elif key == ord(' '):
                paused = not paused
                print("\n" + ("PAUSED" if paused else "RESUMED") + "  ", end='')
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n\nCollection complete!")
        print(f"Total images saved: {count}")
        print(f"Location: {person_dir}")
        print()
    
    def collect_all_people(self, people_names, images_per_person=100):
        """
        Collect facial data for multiple people sequentially.
        
        Args:
            people_names: List of person names
            images_per_person: Number of images to collect per person
        """
        print("\n" + "="*60)
        print("FACIAL DATA COLLECTION")
        print("="*60)
        print(f"People to collect: {', '.join(people_names)}")
        print(f"Images per person: {images_per_person}")
        print("="*60)
        
        for i, person_name in enumerate(people_names, 1):
            print(f"\n[{i}/{len(people_names)}] Ready to collect for: {person_name}")
            input("Press ENTER when ready...")
            
            self.collect_faces(person_name, num_images=images_per_person)
            
            if i < len(people_names):
                print(f"\nMoving to next person in 5 seconds...")
                time.sleep(5)
        
        print("\n" + "="*60)
        print("ALL COLLECTIONS COMPLETE!")
        print("="*60)
        self.print_summary()
    
    def print_summary(self):
        """Print a summary of collected data."""
        print("\nData Collection Summary:")
        print("-" * 60)
        
        if not os.path.exists(self.base_dir):
            print("No data collected yet.")
            return
        
        total_images = 0
        for person_name in os.listdir(self.base_dir):
            person_dir = os.path.join(self.base_dir, person_name)
            if os.path.isdir(person_dir):
                num_images = len([f for f in os.listdir(person_dir) 
                                if f.endswith(('.jpg', '.jpeg', '.png'))])
                print(f"{person_name}: {num_images} images")
                total_images += num_images
        
        print("-" * 60)
        print(f"Total images: {total_images}")
        print(f"Data directory: {os.path.abspath(self.base_dir)}")


def main():
    """Main function to run the face data collection."""
    
    # Configuration
    PEOPLE = ["Dimitri"]  # Change these names
    IMAGES_PER_PERSON = 50  # Adjust as needed
    DATA_DIR = "face_data"  # Directory to save images
    
    # Create collector instance
    collector = FaceDataCollector(base_dir=DATA_DIR)
    
    # Option 1: Collect for all people sequentially
    collector.collect_all_people(PEOPLE, images_per_person=IMAGES_PER_PERSON)
    
    # Option 2: Collect for one person at a time (uncomment to use)
    # collector.collect_faces("person1", num_images=100)


if __name__ == "__main__":
    main()
