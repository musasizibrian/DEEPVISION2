from flask import Flask, render_template, jsonify, request, redirect, url_for, session, g, flash
import os
import json
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import secrets
from flask_session import Session
import mysql.connector  # Import the MySQL connector
import uuid #Use for reset tokens
from dotenv import load_dotenv
import re  # For password strength validation
import pyotp  # Import pyotp
import qrcode  # Import qrcode
import io  # Import io for in-memory image buffers
import base64 # In-memory buffer to base64 image
import hashlib  # Import hashlib for hashing recovery codes

# Initialize Flask application
app = Flask(__name__)

# Load environment variables from .env file
load_dotenv()

app.secret_key = os.environ.get('FLASK_SECRET_KEY') or secrets.token_hex(32)  # Load from env var or generate if not set
app.config['SESSION_TYPE'] = 'filesystem' #Revert back to filesystem.

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
            flash(f"Database Connection Error: {e}", 'error') # Flash an error message
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
        # SQL command to create the users table with is_active column
        create_table_query = """
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(255) NOT NULL,
            email VARCHAR(255) UNIQUE NOT NULL,
            password VARCHAR(255) NOT NULL,
            totp_secret VARCHAR(255),
            recovery_codes TEXT NULL,
            recovery_codes_generated BOOLEAN DEFAULT FALSE,
            role VARCHAR(255) DEFAULT 'viewer',
            is_active BOOLEAN DEFAULT TRUE,
            reset_token VARCHAR(255) DEFAULT NULL,
            reset_token_expiry DATETIME DEFAULT NULL
        );
        """
        cursor.execute(create_table_query)
        db.commit()

    except Exception as e:
        print(f"Error initializing database: {e}")
        db.rollback()
        flash(f"Database Initialization Error: {e}", 'error')

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


def generate_recovery_codes(num_codes=8):
    """Generates a list of unique, secure recovery codes."""
    return [secrets.token_urlsafe(16) for _ in range(num_codes)]

def hash_recovery_code(code):
    """Hashes a recovery code using SHA-256 for secure storage."""
    return hashlib.sha256(code.encode('utf-8')).hexdigest()

def verify_recovery_code(hashed_code, entered_code):
    """Verifies if the entered recovery code matches the stored hashed code."""
    entered_hashed_code = hashlib.sha256(entered_code.encode('utf-8')).hexdigest()
    return hashed_code == entered_hashed_code

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email'].lower()
        password = request.form['password']

        db = get_db()
        if db is None:
            flash("Could not connect to the database.", 'error')
            return render_template('signup.html')
        cursor = db.cursor()

        try:
            cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
            existing_user = cursor.fetchone()
            if existing_user:
                flash("Email address is already registered.", 'error')
                return render_template('signup.html')

            hashed_password = generate_password_hash(password)

            totp_secret = pyotp.random_base32()
            totp_uri = pyotp.TOTP(totp_secret).provisioning_uri(
                name=email,
                issuer_name="DeepVision"
            )

            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=10,
                border=4,
            )
            qr.add_data(totp_uri)
            qr.make(fit=True)

            img = qr.make_image(fill_color="black", back_color="white")
            img_buffer = io.BytesIO()
            img.save(img_buffer, format="PNG")
            img_buffer.seek(0)

            qr_code_base64 = base64.b64encode(img_buffer.read()).decode()

            recovery_codes = generate_recovery_codes()
            hashed_recovery_codes = [hash_recovery_code(code) for code in recovery_codes]
            hashed_recovery_codes_json = json.dumps(hashed_recovery_codes)

            cursor.execute(
                "INSERT INTO users (username, email, password, totp_secret, recovery_codes, recovery_codes_generated, is_active) "
                "VALUES (%s, %s, %s, %s, %s, %s, TRUE)",
                (username, email, hashed_password, totp_secret, hashed_recovery_codes_json, True)
            )

            db.commit()

            session['recovery_codes'] = recovery_codes
            session['qr_code'] = qr_code_base64

            return redirect(url_for('recovery_codes'))

        except mysql.connector.Error as err:
            db.rollback()
            print(f"Error signing up: {err}")
            flash(f"Error signing up: {err}", 'error')
            return render_template('signup.html')

        finally:
            cursor.close()
    return render_template('signup.html')

@app.route('/recovery_codes')
def recovery_codes():
    if 'recovery_codes' in session and 'qr_code' in session:
        recovery_codes = session.pop('recovery_codes')
        qr_code = session.pop('qr_code')
        return render_template('recovery_codes.html', recovery_codes=recovery_codes, qr_code=qr_code)
    else:
        return redirect(url_for('signup'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email'].lower()
        password = request.form['password']
        otp = request.form['otp']
        recovery_code = request.form.get('recovery_code', '')

        db = get_db()
        if db is None:
            flash("Could not connect to the database.", 'error')
            return render_template('login.html')

        cursor = db.cursor(dictionary=True)  # Use dictionary cursor for easier column access
        try:
            cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
            user = cursor.fetchone()

            if user:
                # Simplified active status check - only check is_active column
                if not user.get('is_active', True):  # Default to True if column doesn't exist
                    flash("This account has been deactivated.", 'error')
                    return render_template('login.html')
                
                hashed_password = user['password']
                totp_secret = user['totp_secret']
                hashed_recovery_codes_json = user['recovery_codes']
                recovery_codes_generated = user['recovery_codes_generated']

                if hashed_password is None:
                    flash("Account problem, contact assistance.", 'error')
                    return render_template('login.html')

                if check_password_hash(hashed_password, password):
                    totp = pyotp.TOTP(totp_secret)
                    if totp.verify(otp):
                        session['email'] = email
                        return redirect(url_for('dashboard'))
                    else:
                        if recovery_codes_generated and hashed_recovery_codes_json:
                            hashed_recovery_codes = json.loads(hashed_recovery_codes_json)
                            for hashed_code in hashed_recovery_codes:
                                if verify_recovery_code(hashed_code, recovery_code):
                                    hashed_recovery_codes.remove(hashed_code)
                                    updated_recovery_codes_json = json.dumps(hashed_recovery_codes)

                                    cursor.execute(
                                        "UPDATE users SET recovery_codes = %s, recovery_codes_generated = %s WHERE email = %s",
                                        (updated_recovery_codes_json, False, email)
                                    )
                                    db.commit()
                                    session['email'] = email
                                    flash("Login successful using recovery code. A new set of recovery codes will be generated at next login.", 'success')
                                    return redirect(url_for('dashboard'))
                            flash("Invalid TOTP code or recovery code.", 'error')
                            return render_template('login.html')
                        else:
                            flash("Invalid TOTP code.", 'error')
                            return render_template('login.html')
                else:
                    flash("Invalid email or password.", 'error')
                    return render_template('login.html')
            else:
                flash("Invalid email or password.", 'error')
                return render_template('login.html')
        except mysql.connector.Error as e:
            flash(f"Database Error: {e}", 'error')
            return render_template('login.html')
        finally:
            cursor.close()

    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    session.pop('email', None)
    return redirect(url_for('login'))

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email'].lower()

        db = get_db()
        if db is None:
            return "Could not connect to the database."

        cursor = db.cursor()
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()

        if user:
            reset_token = str(uuid.uuid4())
            reset_token_expiry = datetime.now() + timedelta(hours=1)

            cursor.execute(
                "UPDATE users SET reset_token = %s, reset_token_expiry = %s WHERE email = %s",
                (reset_token, reset_token_expiry, email)
            )
            db.commit()
            return "A password reset link has been sent to your email address."
        else:
            return "Invalid email address."
        db.close()

    return render_template('forgot_password.html')

@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    db = get_db()
    if db is None:
        return "Could not connect to the database."

    cursor = db.cursor()
    cursor.execute("SELECT * FROM users WHERE reset_token = %s AND reset_token_expiry > %s", (token, datetime.now()))
    user = cursor.fetchone()

    if not user:
        cursor.close()
        return "Invalid or expired reset token."

    if request.method == 'POST':
        password = request.form['password']
        hashed_password = generate_password_hash(password)

        cursor.execute(
            "UPDATE users SET password = %s, reset_token = NULL, reset_token_expiry = NULL WHERE email = %s",
            (hashed_password, user[2])
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
def events_page():
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
    new_id = max([event["id"] for event in events]) + 1 if events else 1

    new_event = {
        "id": new_id,
        "time": datetime.now().strftime("%H:%M:%S"),
        "date": "Today",
        "type": event_data.get("type", "Unknown Event"),
        "location": event_data.get("location", "Unknown Location"),
        "severity": event_data.get("severity", "normal")
    }

    events.insert(0, new_event)

    return jsonify(new_event), 201

@app.route('/api/cameras', methods=['GET'])
@login_required
def get_cameras():
    """API endpoint to get all cameras"""
    return jsonify(cameras)  # Uses the dummy cameras data defined earlier

@app.route('/api/camera/<int:camera_id>/feed', methods=['GET'])
@login_required
def get_camera_feed(camera_id):
    """Endpoint to get camera feed (simulated)"""
    camera = next((c for c in cameras if c["id"] == camera_id), None)
    if not camera:
        return jsonify({"error": "Camera not found"}), 404
    
    # In a real implementation, this would return the actual camera feed
    return jsonify({
        "id": camera_id,
        "status": "live" if camera["status"] == "online" else "offline",
        "stream_url": f"/stream/{camera_id}"  # Placeholder for actual stream
    })

@app.route('/api/camera/<int:camera_id>/snapshot', methods=['POST'])
@login_required
def capture_snapshot(camera_id):
    """Endpoint to capture snapshot"""
    # In a real implementation, this would capture and save a snapshot
    return jsonify({
        "status": "success",
        "message": f"Snapshot captured for camera {camera_id}",
        "image_url": f"/snapshots/{camera_id}/{datetime.now().timestamp()}.jpg"
    })

@app.route('/api/camera/<int:camera_id>/record', methods=['POST'])
@login_required
def start_recording(camera_id):
    """Endpoint to start/stop recording"""
    # In a real implementation, this would control recording
    return jsonify({
        "status": "success",
        "message": f"Recording toggled for camera {camera_id}",
        "recording": True  # Would toggle based on current state
    })

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
        new_system_name = request.form.get('systemName')
        new_timezone = request.form.get('timezone')
        new_camera_resolution = request.form.get('cameraResolution')
        new_alert_volume = int(request.form.get('alertVolume'))
        new_theme = request.form.get('theme')

        session['systemName'] = new_system_name
        session['timezone'] = new_timezone
        session['cameraResolution'] = new_camera_resolution
        session['alertVolume'] = new_alert_volume
        session['theme'] = new_theme

        return redirect(url_for('settings'))

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
        return "Could not connect to the database."

    cursor = db.cursor()
    cursor.execute("SELECT id, username, email, role, is_active FROM users")  # Added username to the query
    users = cursor.fetchall()
    cursor.close()

    users_list = []
    for user in users:
        users_list.append({
            "id": user[0],
            "username": user[1],  # Added username
            "email": user[2],
            "role": user[3],
            "is_active": bool(user[4])
        })

    return render_template('user_management.html', users=users_list)
@app.route('/create_user', methods=['GET', 'POST'])
@login_required
def create_user():
    if request.method == 'POST':
        email = request.form['email'].lower()
        password = request.form['password']
        role = request.form['role']
        username = request.form['username']

        db = get_db()
        if db is None:
            flash("Could not connect to the database.", 'error')
            return render_template('create_user.html')

        cursor = db.cursor()

        try:
            cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
            existing_user = cursor.fetchone()
            if existing_user:
                flash("Email address is already registered.", 'error')
                return render_template('create_user.html')

            hashed_password = generate_password_hash(password)

            cursor.execute(
                "INSERT INTO users (username, email, password, role, is_active) VALUES (%s, %s, %s, %s, TRUE)",
                (username, email, hashed_password, role)
            )
            db.commit()
            flash("User created successfully!", 'success')
            return redirect(url_for('user_management'))

        except mysql.connector.Error as err:
            db.rollback()
            print(f"Error creating user: {err}")
            flash(f"Error creating user: {err}", 'error')
            return render_template('create_user.html')

        finally:
            cursor.close()

    return render_template('create_user.html')

@app.route('/edit_user/<int:user_id>', methods=['GET', 'POST'])
@login_required
def edit_user(user_id):
    db = get_db()
    if db is None:
        return "Could not connect to the database."

    cursor = db.cursor()

    if request.method == 'POST':
        username = request.form['username']
        role = request.form['role']
        email = request.form['email']

        try:
            cursor.execute(
                "UPDATE users SET username = %s, role = %s, email = %s WHERE id = %s",
                (username, role, email, user_id)
            )
            db.commit()
            flash("User updated successfully!", 'success')
            return redirect(url_for('user_management'))
        except mysql.connector.Error as err:
            db.rollback()
            flash(f"Error updating user: {err}", 'error')

    cursor.execute("SELECT id, username, email, role, is_active FROM users WHERE id = %s", (user_id,))
    user = cursor.fetchone()
    cursor.close()

    if user:
        user_data = {
            "id": user[0],
            "username": user[1],
            "email": user[2],
            "role": user[3],
            "is_active": bool(user[4])
        }
        return render_template('edit_user.html', user=user_data)
    else:
        return "User not found."

@app.route('/deactivate_user/<int:user_id>')
@login_required
def deactivate_user(user_id):
    db = get_db()
    if db is None:
        flash("Could not connect to the database.", 'error')
        return redirect(url_for('user_management'))

    cursor = db.cursor()

    try:
        cursor.execute("UPDATE users SET is_active = FALSE WHERE id = %s", (user_id,))
        db.commit()
        flash("User deactivated successfully!", 'success')
    except mysql.connector.Error as err:
        db.rollback()
        flash(f"Error deactivating user: {err}", 'error')
    finally:
        cursor.close()

    return redirect(url_for('user_management'))

@app.route('/reactivate_user/<int:user_id>')
@login_required
def reactivate_user(user_id):
    db = get_db()
    if db is None:
        flash("Could not connect to the database.", 'error')
        return redirect(url_for('user_management'))

    cursor = db.cursor()

    try:
        cursor.execute("UPDATE users SET is_active = TRUE WHERE id = %s", (user_id,))
        db.commit()
        flash("User reactivated successfully!", 'success')
    except mysql.connector.Error as err:
        db.rollback()
        flash(f"Error reactivating user: {err}", 'error')
    finally:
        cursor.close()

    return redirect(url_for('user_management'))

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
        with open(placeholder_path, 'w') as f:
            f.write('placeholder')

    with app.app_context():
        init_db()
    # Run the Flask application
    app.run(debug=True, host='0.0.0.0', port=5000)