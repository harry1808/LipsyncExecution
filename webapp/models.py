from datetime import datetime

from flask_login import UserMixin
from werkzeug.security import check_password_hash, generate_password_hash

from . import db


class User(UserMixin, db.Model):
    """Registered application user."""

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f"<User {self.username}>"


class Activity(db.Model):
    """Stores user dubbing jobs and outcomes."""

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    output_filename = db.Column(db.String(255), nullable=True)
    stored_original_filename = db.Column(db.String(255), nullable=True)
    source_lang = db.Column(db.String(8), nullable=False)
    dest_lang = db.Column(db.String(8), nullable=False)
    transcript = db.Column(db.Text)
    translated_text = db.Column(db.Text)
    status = db.Column(db.String(32), default="pending")
    error_message = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship("User", backref=db.backref("activities", lazy=True))

    def mark_finished(self, output_filename, transcript, translated_text, original_filename=None):
        self.output_filename = output_filename
        self.transcript = transcript
        self.translated_text = translated_text
        self.status = "completed"
        if original_filename:
            self.stored_original_filename = original_filename

    def mark_failed(self, error_message):
        self.status = "failed"
        self.error_message = error_message

    def __repr__(self):
        return f"<Activity {self.id} ({self.status})>"

