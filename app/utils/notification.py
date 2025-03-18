# app/utils/notification.py
import datetime

class NotificationSystem:
    def __init__(self):
        self.alerts = []
    
    def generate_alert(self, threat_type, confidence, location):
        """Generate an alert based on detected threat"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        alert = {
            "timestamp": timestamp,
            "threat_type": threat_type,
            "confidence": confidence,
            "location": location
        }
        self.alerts.append(alert)
        return alert
    
    def get_recent_alerts(self, count=10):
        """Get the most recent alerts"""
        return self.alerts[-count:] if self.alerts else []