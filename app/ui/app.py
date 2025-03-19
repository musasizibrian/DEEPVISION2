from flask import Flask, render_template, jsonify, request, redirect, url_for, session, g
import os
import json
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash, check_password_hash
import secrets
import smtplib
from flask_session import Session
import mysql.connector
from email.mime.text import MIMEText
import uuid
from dotenv import load_dotenv
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField
from wtforms.validators import InputRequired, Email

# Initialize Flask application
app = Flask(__name__)

# Load environment variables from .env file
load_dotenv()

app.secret_key = os.environ.get('FLASK_SECRET_KEY') or secrets.token_hex(32)  # Load from env var or generate if not set
app.config['SESSION_TYPE'] = 'filesystem'

# Email Configuration
app.config['MAIL_SERVER'] = os.environ.get('MAIL_SERVER') or 'smtp.gmail.com'
app.config['MAIL_PORT'] = int(os.environ.get('MAIL_PORT') or 587)
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME') or 'your-email@gmail.com'
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD')
app.config['MAIL_FROM_ADDRESS'] = os.environ.get('MAIL_FROM_ADDRESS') or 'your-email@gmail.com'

# MySQL Configuration
app.config['MYSQL_HOST'] = os.environ.get("MYSQL_HOST") or 'localhost'
app.config['MYSQL_USER'] = os.environ.get("MYSQL_USER") or 'root'
app.config['MYSQL_PASSWORD'] = os.environ.get("MYSQL_PASSWORD") or ''
app.config['MYSQL_DB'] = os.environ.get("MYSQL_DB") or 'appdb'

# Flask form for signup using Flask-WTF
class SignupForm(FlaskForm):
    email = StringField('Email', validators=[InputRequired(), Email()])
    password = PasswordField('Password', validators=[InputRequired()])

# Connect to the database
def get_db():
    if 'db' not in g:
        try:
            g.db = mysql.connector.connect(
                host=app.config['MYSQL_HOST'],
                user=app.config['MYSQL_USER'],
                password=app.config['MYSQL_PASSWORD'],
                database=app.config['MYSQL_DB'],
                autocommit=True,
            )
        except mysql.connector.Error as e:
            print(f"Error connecting to MySQL: {e}")
            return None
    return g.db

@app.teardown_appcontext
def close_db(error=None):
    db = getattr(g, 'db', None)
    if db is not None and db.is_connected():
        db.close()

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    form = SignupForm()
    if request.method == 'POST' and form.validate_on_submit():
        email = form.email.data
        password = form.password.data
        hashed_password = generate_password_hash(password)

        db = get_db()
        if db is None:
            return "Could not connect to the database."  # Error message
        cursor = db.cursor()

        try:
            cursor.execute("INSERT INTO users (email, password) VALUES (%s, %s)", (email, hashed_password))
            db.commit()
            return redirect(url_for('login'))

        except mysql.connector.Error as err:
            db.rollback()
            return f"Error signing up: {err}"

        finally:
            cursor.close()

    return render_template('signup.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        db = get_db()
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
def logout():
    session.pop('email', None)
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
