from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import os

# Create a Flask application instance
app = Flask(__name__)

# Database configuration
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'users.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy
db = SQLAlchemy(app)

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

    def __repr__(self):
        return f'<User {self.username}>'

# View data within the app context
with app.app_context():
    try:
        db.create_all()  # Ensure table exists (optional, remove if already created)
        users = User.query.all()
        if users:
            for user in users:
                print(f"ID: {user.id}, Username: {user.username}, Email: {user.email}, Password: {user.password}")
        else:
            print("No users found in the database.")
    except Exception as e:
        print(f"Error accessing database: {e}")