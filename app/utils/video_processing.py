# app/utils/video_processing.py
import cv2
import numpy as np
import threading
import time
from PIL import Image

class VideoProcessor:
    def __init__(self, classification_model, notification_system):
        self.classification_model = classification_model
        self.notification_system = notification_system
        self.cameras = {}  # Store camera objects
        self.processing_threads = {}  # Store processing threads
        
    # (Rest of the methods remain the same as in your original class)
    
    def _analyze_frame(self, frame, camera_id, location):
        """Analyze a frame for classification"""
        processed_frame = frame.copy()
        
        # Convert BGR to RGB (OpenCV uses BGR, PIL uses RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Classify the frame
        result = self.classification_model.classify(rgb_frame)
        class_name = result["class"]
        confidence = result["confidence"]
        
        # Add classification label to the frame
        label = f"{class_name}: {confidence:.2f}"
        cv2.putText(processed_frame, label, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Generate alerts for non-normal scenes
        if class_name != "Normal" and confidence > 0.6:  # Threshold can be adjusted
            self.notification_system.generate_alert(
                class_name.lower(), 
                confidence, 
                f"Camera {camera_id}: {location}"
            )
            
            # Add visual indicator for alerting scenes
            if class_name == "Violence":
                cv2.rectangle(processed_frame, (0, 0), (processed_frame.shape[1], processed_frame.shape[0]), 
                             (0, 0, 255), 5)  # Red border for Violence
            elif class_name == "Weaponized":
                cv2.rectangle(processed_frame, (0, 0), (processed_frame.shape[1], processed_frame.shape[0]), 
                             (255, 0, 0), 5)  # Blue border for Weaponized
        
        return processed_frame