from flask import Flask, render_template, jsonify, request, redirect, url_for, session, g
import os
import json
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import secrets
import smtplib
from flask_session import Session
import mysql.connector  # Import the MySQL connector
from email.mime.text import MIMEText
#from supabase import create_client, Client #Remove
import uuid #Use for reset tokens
from dotenv import load_dotenv

# Initialize Flask application
app = Flask(__name__)

# Load environment variables from .env file
load_dotenv()

app.secret_key = os.environ.get('FLASK_SECRET_KEY') or secrets.token_hex(32)  # Load from env var or generate if not set
#app.config['SESSION_TYPE'] = 'redis' #Comment out or remove this line
#app.config['SESSION_REDIS'] = redis.Redis.from_url(os.environ.get("REDIS_URL")) #Comment out or remove this line
app.config['SESSION_TYPE'] = 'filesystem' #Revert back to filesystem.

# Email Configuration - VERY IMPORTANT: Configure this properly
app.config['MAIL_SERVER'] = os.environ.get('MAIL_SERVER') or 'smtp.gmail.com'  # Replace with your SMTP server
app.config['MAIL_PORT'] = int(os.environ.get('MAIL_PORT') or 587)  # Replace with your SMTP port
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME') or 'makobisimon@gmail.com'  # Replace with your email address
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD') # Replace with your email password
app.config['MAIL_FROM_ADDRESS'] = os.environ.get('MAIL_FROM_ADDRESS') or 'makobisimon@gmail.com' #Replace with your email address

# MySQL Configuration
app.config['MYSQL_HOST'] = os.environ.get("MYSQL_HOST") or 'localhost'  # Default to localhost if not set
app.config['MYSQL_USER'] = os.environ.get("MYSQL_USER") or 'root'  # Replace with your MySQL username
app.config['MYSQL_PASSWORD'] = os.environ.get("MYSQL_PASSWORD") or ''  # Replace with your MySQL password
app.config['MYSQL_DB'] = os.environ.get("MYSQL_DB") or 'appdb'  # Replace with your MySQL database name

# Dummy data for events
events = [
    {
        "id": 1,
        "time": "10:30:00 AM",
        "date": "Today",
        "type": "Suspicious Activity",
        "location": "Camera 2 - Kampala Road Camera 1"
    },
    {
        "id": 2,
        "time": "09:52:10 AM",
        "date": "Today",
        "type": "Person of Interest Identified",
        "location": "Camera 1 - City Hall Main Entrance"
    },
    {
        "id": 3,
        "time": "09:52:10 AM",
        "date": "Today",
        "type": "Weapon Detected (Gun)",
        "location": "Camera 1 - Nakasero Roundabout"
    },
    {
        "id": 4,
        "time": "08:00:00 AM",
        "date": "Today",
        "type": "System Started",
        "location": "Monitoring Server"
    },
    {
        "id": 5,
        "time": "08:05:00 AM",
        "date": "Today",
        "type": "System Scan Complete",
        "location": "Monitoring Server"
    }
]

# Dummy data for cameras
cameras = [
    {"id": 1, "location": "City Hall Main Entrance", "status": "online"},
    {"id": 2, "location": "Kampala Road Camera 1", "status": "online", "alert": "Suspicious Activity"},
    {"id": 3, "location": "Nakasero Roundabout", "status": "online"},
    {"id": 4, "location": "Kampala Road Camera 2", "status": "online"},
    {"id": 5, "location": "Central Police Station", "status": "online"},
    {"id": 6, "location": "Centenary Bank Entrance", "status": "online"},
    {"id": 7, "location": "Nakasero Market", "status": "offline"},
    {"id": 8, "location": "Parliament Avenue", "status": "offline"}
]

#Dummy data for reports
reports = [
    {
        "title": "Daily Activity Report",
        "description": "Summary of all activity recorded on the system for the day.",
        "date": "2024-10-27"
    },
    {
        "title": "Weekly Security Audit",
        "description": "Detailed audit of security events and system performance over the past week.",
        "date": "2024-10-26"
    },
    {
        "title": "Monthly System Health Report",
        "description": "Overview of system health metrics and performance for the month.",
        "date": "2024-10-20"
    }
]

#Dummy data for settings
default_settings = {
    "systemName": "DeepVision Monitoring Center",
    "timezone": "UTC",
    "cameraResolution": "720p",
    "alertVolume": 70,
    "theme": "light"
}

def get_db():
    if 'db' not in g:
        try:
            g.db = mysql.connector.connect(
                host=app.config['MYSQL_HOST'],
                user=app.config['MYSQL_USER'],
                password=app.config['MYSQL_PASSWORD'],
                database=app.config['MYSQL_DB'],
                autocommit=True, # enable autocommit
            )
        except mysql.connector.Error as e:
            print(f"Error connecting to MySQL: {e}")
            return None  # Return None if the connection fails
    return g.db

@app.teardown_appcontext
def close_db(error=None):
    db = getattr(g, 'db', None)
    if db is not None and db.is_connected():
        db.close()

def init_db():
    """Creates the database tables."""
    db = get_db()
    if db is None:
        print("Failed to connect to the database, init_db skipped.")
        return  # Exit if the database connection failed
    cursor = db.cursor()

    try:
        with open('schema.sql', 'r') as f:
            sql_commands = f.read().split(';')
            for command in sql_commands:
                if command.strip():  # Execute each SQL command
                    cursor.execute(command)
        db.commit()  # Commit the changes

    except Exception as e:
        print(f"Error initializing database: {e}")
        db.rollback()  # Rollback changes if an error occurs

    finally:
        if db.is_connected():
            cursor.close()
    print("Database initialized successfully.")


# Authentication decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'email' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def send_reset_email(email, token):
    """Sends a password reset email to the user."""
    reset_url = url_for('reset_password', token=token, _external=True)
    subject = "Password Reset Request"
    body = f"Please click this link to reset your password: {reset_url}"

    msg = MIMEText(body, 'plain')
    msg['Subject'] = subject
    msg['From'] = app.config['MAIL_FROM_ADDRESS']
    msg['To'] = email

    try:
        with smtplib.SMTP(app.config['MAIL_SERVER'], app.config['MAIL_PORT']) as server:
            server.starttls()
            server.login(app.config['MAIL_USERNAME'], app.config['MAIL_PASSWORD'])
            server.sendmail(app.config['MAIL_FROM_ADDRESS'], [email], msg.as_string())
        print(f"Password reset email sent to {email}")
    except Exception as e:
        print(f"Error sending email: {e}")
        # Handle the exception appropriately, e.g., log it, display an error message to the user


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        # Hash the password before storing it
        hashed_password = generate_password_hash(password)

        db = get_db() #To get the database from function call and not directly because may not exist with database already setup.
        if db is None:
            return "Could not connect to the database."  # Error message
        cursor = db.cursor()

        try:
            cursor.execute("INSERT INTO users (email, password) VALUES (%s, %s)", (email, hashed_password))
            db.commit()
            return redirect(url_for('login'))

        except mysql.connector.Error as err:
            db.rollback()
            print(f"Error signing up: {err}")
            return f"Error signing up: {err}"

        finally:
            cursor.close()

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        db = get_db()  # Get the database connection
        if db is None:
            return "Could not connect to the database."  # Error message

        cursor = db.cursor()
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()

        if user and check_password_hash(user[2], password):
            session['email'] = email
            return redirect(url_for('dashboard'))
        else:
            return "Invalid email or password"

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    session.pop('email', None)
    return redirect(url_for('login'))

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email']

        db = get_db()
        if db is None:
            return "Could not connect to the database."  # Error message

        cursor = db.cursor()
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()

        if user:
            # Generate a unique reset token
            reset_token = str(uuid.uuid4())
            # Set the expiry time for the token (e.g., 1 hour from now)
            reset_token_expiry = datetime.now() + timedelta(hours=1)

            # Update the user's record with the reset token and expiry
            cursor.execute(
                "UPDATE users SET reset_token = %s, reset_token_expiry = %s WHERE email = %s",
                (reset_token, reset_token_expiry, email)
            )
            db.commit()
            # Send the reset email
            send_reset_email(email, reset_token)
            return "A password reset link has been sent to your email address."
        else:
            return "Invalid email address."
        db.close()

    return render_template('forgot_password.html')

@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    db = get_db()
    if db is None:
        return "Could not connect to the database."  # Error message

    cursor = db.cursor()
    cursor.execute("SELECT * FROM users WHERE reset_token = %s AND reset_token_expiry > %s", (token, datetime.now()))
    user = cursor.fetchone()

    if not user:
        cursor.close()
        return "Invalid or expired reset token."

    if request.method == 'POST':
        password = request.form['password']
        hashed_password = generate_password_hash(password)

        # Update the user's password and clear the reset token
        cursor.execute(
            "UPDATE users SET password = %s, reset_token = NULL, reset_token_expiry = NULL WHERE email = %s",
            (hashed_password, user[1]) #1 is the email if the order is id/email/password
        )
        db.commit()
        cursor.close()
        return redirect(url_for('login'))

    return render_template('reset_password.html', token=token)

@app.route('/')
def root():
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
@login_required
def dashboard():
    """Render the main dashboard page"""
    return render_template('index.html')

@app.route('/alerts')
@login_required
def alerts():
    """Render the alerts page"""
    return render_template('alerts.html')

@app.route('/events')
@login_required
def events_page():  # Changed function name to avoid conflict with the events list
    """Render the events page"""
    return render_template('events.html', events=events)

@app.route('/reports')
@login_required
def reports():
    """Render the reports page"""
    return render_template('reports.html', reports=reports)

@app.route('/api/events', methods=['GET'])
@login_required
def get_events():
    """API endpoint to get all events"""
    return jsonify(events)

@app.route('/api/events', methods=['POST'])
@login_required
def add_event():
    """API endpoint to add a new event"""
    event_data = request.json
    # Generate a new ID (in a real app, this would be handled by a database)
    new_id = max([event["id"] for event in events]) + 1 if events else 1

    # Create the new event
    new_event = {
        "id": new_id,
        "time": datetime.now().strftime("%H:%M:%S"),
        "date": "Today",
        "type": event_data.get("type", "Unknown Event"),
        "location": event_data.get("location", "Unknown Location"),
        "severity": event_data.get("severity", "normal")
    }

    # Add to events list (in a real app, this would be added to a database)
    events.insert(0, new_event)

    return jsonify(new_event), 201

@app.route('/api/status', methods=['GET'])
@login_required
def get_status():
    """API endpoint to get system status"""
    return jsonify(events)

@app.route('/api/cameras', methods=['GET'])
@login_required
def get_cameras():
    """API endpoint to get all cameras"""
    return jsonify(cameras)

@app.route('/cameras')
@login_required
def cameras():
    return render_template('all_cameras.html')

@app.route('/api/cameras/<int:camera_id>/alert', methods=['POST'])
@login_required
def set_camera_alert(camera_id):
    """API endpoint to set an alert on a camera"""
    alert_data = request.json

    # Find the camera with the given ID
    for camera in cameras:
        if camera["id"] == camera_id:
            camera["alert"] = alert_data.get("alert")
            return jsonify(camera)

    return jsonify({"error": "Camera not found"}), 404

@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    """Render the settings page and handle settings updates."""
    if request.method == 'POST':
        # Get the data from the submitted form
        new_system_name = request.form.get('systemName')
        new_timezone = request.form.get('timezone')
        new_camera_resolution = request.form.get('cameraResolution')
        new_alert_volume = int(request.form.get('alertVolume'))
        new_theme = request.form.get('theme')

        # Update the settings.  In a real app, you'd save these to a database or config file.
        session['systemName'] = new_system_name
        session['timezone'] = new_timezone
        session['cameraResolution'] = new_camera_resolution
        session['alertVolume'] = new_alert_volume
        session['theme'] = new_theme

        return redirect(url_for('settings'))  # Redirect to refresh the page

    # Get the current settings. If they don't exist in the session, use the defaults.
    current_settings = {
        'systemName': session.get('systemName', default_settings['systemName']),
        'timezone': session.get('timezone', default_settings['timezone']),
        'cameraResolution': session.get('cameraResolution', default_settings['cameraResolution']),
        'alertVolume': session.get('alertVolume', default_settings['alertVolume']),
        'theme': session.get('theme', default_settings['theme'])
    }

    return render_template('settings.html', settings=current_settings)


@app.route('/user_management')
@login_required
def user_management():
    db = get_db()
    if db is None:
        return "Could not connect to the database."  # Error message
    cursor = db.cursor()
    cursor.execute("SELECT id, email, password FROM users")
    users = cursor.fetchall()
    cursor.close()

    # Convert to list of dictionaries for easier template rendering
    users_list = []
    for user in users:
        # You might want to fetch role information from another table based on user ID
        role = "viewer"  # Default role, adjust as needed
        users_list.append({"id": user[0], "email": user[1],  "role": role}) #password column removed

    return render_template('user_management.html', users=users_list)


@app.route('/create_user')
@login_required
def create_user():
    return "This URL will be reponsible for the user creation"

@app.route('/edit_user/<user_id>')
@login_required
def edit_user(user_id):
    return f"This URL will be reponsible for editing {user_id}"

@app.route('/deactivate_user/<user_id>')
@login_required
def deactivate_user(user_id):
    return f"This URL will be reponsible for deactivating {user_id}"
@app.route('/system_status')
@login_required
def system_status():
    """Remove comment on these to load this function"""
    return render_template('system_status.html')

@app.route('/api/reports', methods=['GET'])
@login_required
def get_reports():
    """API endpoint to get all reports"""
    return jsonify(reports)

if __name__ == '__main__':
    # Create directories for static files if they don't exist
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    os.makedirs('static/images', exist_ok=True)

    # Create a placeholder image if it doesn't exist
    placeholder_path = 'static/images/placeholder.jpg'
    if not os.path.exists(placeholder_path):
        # This would create a blank image - in a real app, you'd want to provide an actual placeholder
        with open(placeholder_path, 'w') as f:
            f.write('placeholder')

    with app.app_context():
        init_db()
    # Run the Flask application
    app.run(debug=True, host='0.0.0.0', port=5000)