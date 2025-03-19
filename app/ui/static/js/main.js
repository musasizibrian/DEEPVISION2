// DeepVision Monitoring Center Interactive Features

document.addEventListener('DOMContentLoaded', function() {
    // Initialize interactive elements
    initializeEventFilters();
    initializeCameraAlerts();
    initializeButtonActions();
    initializeSimulatedLiveFeeds();
  });
  
  // Event filtering functionality
  function initializeEventFilters() {
    const filterButtons = document.querySelectorAll('.events-filter button');
    const eventItems = document.querySelectorAll('.event-item');
    
    filterButtons.forEach(button => {
      button.addEventListener('click', function() {
        // Remove active class from all buttons
        filterButtons.forEach(btn => btn.classList.remove('active-filter'));
        
        // Add active class to clicked button
        this.classList.add('active-filter');
        
        const filterType = this.textContent.toLowerCase();
        
        // Show all events if "All" is selected
        if (filterType === 'all') {
          eventItems.forEach(item => item.style.display = 'grid');
          return;
        }
        
        // Filter events based on button text
        eventItems.forEach(item => {
          const eventType = determineEventType(item);
          if (filterType === 'alerts' && (eventType === 'weapon' || eventType === 'suspicious')) {
            item.style.display = 'grid';
          } else if (filterType === 'activities' && (eventType === 'person' || eventType === 'system')) {
            item.style.display = 'grid';
          } else {
            item.style.display = 'none';
          }
        });
      });
    });
  }
  
  // Helper function to determine event type from icon class
  function determineEventType(eventItem) {
    const iconDiv = eventItem.querySelector('.event-icon');
    if (iconDiv.classList.contains('weapon')) return 'weapon';
    if (iconDiv.classList.contains('suspicious')) return 'suspicious';
    if (iconDiv.classList.contains('person')) return 'person';
    if (iconDiv.classList.contains('system')) return 'system';
    return 'unknown';
  }
  
  // Camera alerts functionality
  function initializeCameraAlerts() {
    // Add click event to camera cards to show details
    const cameraCards = document.querySelectorAll('.camera-card');
    
    cameraCards.forEach(card => {
      card.addEventListener('click', function() {
        const cameraTitle = this.querySelector('.camera-title span').textContent;
        const cameraLocation = this.querySelector('.camera-location').textContent;
        const hasAlert = this.querySelector('.alert') !== null;
        
        let alertMessage = hasAlert 
          ? `Alert detected on ${cameraTitle} (${cameraLocation})!` 
          : `${cameraTitle} (${cameraLocation}) - No alerts`;
          
        showNotification(alertMessage, hasAlert ? 'danger' : 'info');
      });
    });
  }
  
  // Button actions
  function initializeButtonActions() {
    // Settings and logout buttons
    const settingsBtn = document.querySelector('.user-controls .btn-outline');
    const logoutBtn = document.querySelector('.user-controls .btn-primary');
    
    settingsBtn.addEventListener('click', function() {
      showNotification('Settings panel would open here', 'info');
    });
    
    logoutBtn.addEventListener('click', function() {
      if (confirm('Are you sure you want to logout?')) {
        showNotification('Logging out...', 'info');
        // Simulate logout delay
        setTimeout(() => {
          // In a real app this would redirect to login page
          showNotification('You have been logged out', 'success');
        }, 1500);
      }
    });
    
    // Event view buttons
    const viewButtons = document.querySelectorAll('.event-actions button');
    viewButtons.forEach(button => {
      button.addEventListener('click', function(e) {
        e.stopPropagation(); // Prevent event from bubbling up
        const eventDetails = this.closest('.event-item').querySelector('.event-details div:first-child').textContent;
        showNotification(`Viewing details: ${eventDetails}`, 'info');
      });
    });
    
    // Navigation links
    const navLinks = document.querySelectorAll('.nav-link');
    navLinks.forEach(link => {
      link.addEventListener('click', function(e) {
        e.preventDefault();
        
        // Remove active class from all links
        navLinks.forEach(l => l.classList.remove('active'));
        
        // Add active class to clicked link
        this.classList.add('active');
        
        const pageName = this.textContent.trim();
        showNotification(`Navigating to ${pageName} page`, 'info');
      });
    });
  }
  
  // Simulated live feeds with occasional alerts
  function initializeSimulatedLiveFeeds() {
    const cameraFeeds = document.querySelectorAll('.camera-feed');
    
    // Simulate activity on cameras
    setInterval(() => {
      const randomCamera = Math.floor(Math.random() * cameraFeeds.length);
      const camera = cameraFeeds[randomCamera];
      
      // Check if this camera already has an alert
      const existingAlert = camera.querySelector('.alert');
      if (existingAlert) {
        // 30% chance to remove existing alert
        if (Math.random() < 0.3) {
          existingAlert.remove();
          const cameraTitle = camera.closest('.camera-card').querySelector('.camera-title span').textContent;
          showNotification(`Alert cleared on ${cameraTitle}`, 'success');
        }
      } else {
        // 15% chance to add a new alert
        if (Math.random() < 0.15) {
          const alertTypes = [
            { text: 'Suspicious Activity', class: 'suspicious' },
            { text: 'Weapon Detected', class: 'weapon' }
          ];
          
          const alertType = alertTypes[Math.floor(Math.random() * alertTypes.length)];
          const alertDiv = document.createElement('div');
          alertDiv.className = 'alert';
          alertDiv.textContent = alertType.text;
          camera.appendChild(alertDiv);
          
          const cameraTitle = camera.closest('.camera-card').querySelector('.camera-title span').textContent;
          showNotification(`New alert on ${cameraTitle}: ${alertType.text}`, 'danger');
          
          // Add to events list
          addNewEvent(alertType.class, `${alertType.text} at ${camera.closest('.camera-card').querySelector('.camera-location').textContent}`);
        }
      }
    }, 8000); // Check every 8 seconds
    
    // Simulate camera feed flicker occasionally
    setInterval(() => {
      const randomCamera = Math.floor(Math.random() * cameraFeeds.length);
      const camera = cameraFeeds[randomCamera];
      
      // Add a quick flicker effect
      camera.classList.add('flicker');
      setTimeout(() => {
        camera.classList.remove('flicker');
      }, 500);
    }, 15000); // Every 15 seconds
  }
  
  // Add new event to the events list
  function addNewEvent(type, text) {
    const eventsList = document.querySelector('.events-list');
    const newEvent = document.createElement('div');
    newEvent.className = 'event-item';
    
    // Get current time
    const now = new Date();
    const hours = now.getHours();
    const minutes = now.getMinutes();
    const seconds = now.getSeconds();
    const timeString = `${hours}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')} ${hours >= 12 ? 'PM' : 'AM'}`;
    
    // Determine icon based on type
    let iconSvg = '';
    if (type === 'suspicious') {
      iconSvg = '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>';
    } else if (type === 'weapon') {
      iconSvg = '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78L12 21.23l8.84-8.84a5.5 5.5 0 0 0 0-7.78z"></path></svg>';
    }
    
    newEvent.innerHTML = `
      <div class="event-icon ${type}">
        ${iconSvg}
      </div>
      <div class="event-details">
        <div>${text}</div>
        <div class="timestamp">${timeString} - Today</div>
      </div>
      <div class="event-actions">
        <button class="btn-outline">View</button>
      </div>
    `;
    
    // Add event to the top of the list
    if (eventsList.firstChild) {
      eventsList.insertBefore(newEvent, eventsList.firstChild);
    } else {
      eventsList.appendChild(newEvent);
    }
    
    // Remove oldest event if there are more than 10
    const events = eventsList.querySelectorAll('.event-item');
    if (events.length > 10) {
      eventsList.removeChild(events[events.length - 1]);
    }
    
    // Add click handler to the new view button
    const viewButton = newEvent.querySelector('.event-actions button');
    viewButton.addEventListener('click', function(e) {
      e.stopPropagation();
      showNotification(`Viewing details: ${text}`, 'info');
    });
  }
  
  // Show notification function
  function showNotification(message, type = 'info') {
    // Create notification container if it doesn't exist
    let notificationContainer = document.querySelector('.notification-container');
    if (!notificationContainer) {
      notificationContainer = document.createElement('div');
      notificationContainer.className = 'notification-container';
      document.body.appendChild(notificationContainer);
      
      // Add styles for notifications
      const style = document.createElement('style');
      style.textContent = `
        .notification-container {
          position: fixed;
          top: 20px;
          right: 20px;
          z-index: 1000;
          display: flex;
          flex-direction: column;
          gap: 10px;
        }
        
        .notification {
          padding: 12px 20px;
          border-radius: 4px;
          color: white;
          box-shadow: 0 3px 10px rgba(0,0,0,0.15);
          transform: translateX(120%);
          transition: transform 0.3s ease;
          display: flex;
          align-items: center;
          justify-content: space-between;
          min-width: 280px;
        }
        
        .notification.show {
          transform: translateX(0);
        }
        
        .notification.info {
          background-color: var(--primary);
        }
        
        .notification.success {
          background-color: var(--success);
        }
        
        .notification.danger {
          background-color: var(--danger);
        }
        
        .notification-close {
          background: none;
          border: none;
          color: white;
          cursor: pointer;
          font-weight: bold;
          font-size: 16px;
          margin-left: 10px;
        }
        
        .camera-feed {
          transition: opacity 0.05s;
        }
        
        .camera-feed.flicker {
          opacity: 0.7;
        }
        
        .events-filter button.active-filter {
          background-color: var(--primary);
          color: white;
        }
      `;
      document.head.appendChild(style);
    }
    
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.innerHTML = `
      ${message}
      <button class="notification-close">&times;</button>
    `;
    
    notificationContainer.appendChild(notification);
    
    // Show the notification with a small delay
    setTimeout(() => {
      notification.classList.add('show');
    }, 10);
    
    // Add close button functionality
    const closeButton = notification.querySelector('.notification-close');
    closeButton.addEventListener('click', function() {
      notification.classList.remove('show');
      setTimeout(() => {
        notificationContainer.removeChild(notification);
      }, 300);
    });
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
      if (notificationContainer.contains(notification)) {
        notification.classList.remove('show');
        setTimeout(() => {
          if (notificationContainer.contains(notification)) {
            notificationContainer.removeChild(notification);
          }
        }, 300);
      }
    }, 5000);
  }