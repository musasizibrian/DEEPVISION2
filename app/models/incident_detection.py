# app/models/incident_detection.py
import cv2
import numpy as np
from collections import deque

class IncidentDetectionModel:
    def __init__(self, model_path=None, frame_history=10):
        # For a more advanced implementation, you could use action recognition models
        # For now, let's implement motion-based detection
        self.frame_history = deque(maxlen=frame_history)
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True)
        
    def detect(self, frame):
        """Detect suspicious activities in a frame"""
        # Apply background subtraction
        fg_mask = self.background_subtractor.apply(frame)
        
        # Add frame to history
        self.frame_history.append(frame.copy())
        
        # If we don't have enough frames yet, return no incidents
        if len(self.frame_history) < self.frame_history.maxlen:
            return []
        
        # Calculate motion metrics
        motion_score = self._calculate_motion_score(fg_mask)
        
        # Detect rapid motion or unusual patterns
        incidents = []
        if motion_score > 50:  # Threshold for suspicious activity
            incidents.append({
                "type": "suspicious_activity",
                "confidence": motion_score / 100,
                "bbox": self._get_motion_area(fg_mask)
            })
        
        return incidents
        
    def _calculate_motion_score(self, fg_mask):
        """Calculate a score representing the amount of motion"""
        motion_pixels = cv2.countNonZero(fg_mask)
        total_pixels = fg_mask.shape[0] * fg_mask.shape[1]
        motion_percentage = (motion_pixels / total_pixels) * 100
        return motion_percentage
        
    def _get_motion_area(self, fg_mask):
        """Get bounding box of motion area"""
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return [0, 0, 0, 0]
            
        # Get bounding box of the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        return [x, y, x+w, y+h]