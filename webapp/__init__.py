import os
from pathlib import Path

import random

from flask import Flask
from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()
login_manager = LoginManager()
login_manager.login_view = "auth.login"


def create_app():
    """Application factory for the Flask UI."""
    app = Flask(__name__, instance_relative_config=True)

    instance_path = Path(app.instance_path)
    instance_path.mkdir(parents=True, exist_ok=True)

    upload_dir = instance_path / "uploads"
    output_dir = instance_path / "outputs"
    upload_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    background_images = [
        "https://images.unsplash.com/photo-1470229538611-16ba8c7ffbd7?auto=format&fit=crop&w=2000&q=80",
        "https://images.unsplash.com/photo-1517414204284-519f67b38720?auto=format&fit=crop&w=2000&q=80",
        "https://images.unsplash.com/photo-1469474968028-56623f02e42e?auto=format&fit=crop&w=2000&q=80",
        "https://images.unsplash.com/photo-1484704849700-f032a568e944?auto=format&fit=crop&w=2000&q=80",
    ]

    app.config.from_mapping(
        SECRET_KEY=os.environ.get("FLASK_SECRET_KEY", "replace-me"),
        SQLALCHEMY_DATABASE_URI=f"sqlite:///{instance_path / 'app.db'}",
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        UPLOAD_FOLDER=str(upload_dir),
        OUTPUT_FOLDER=str(output_dir),
        MAX_CONTENT_LENGTH=1024 * 1024 * 1024,  # 1 GB upload ceiling
        NLLB_MODEL_NAME=os.environ.get(
            "NLLB_MODEL_NAME", "facebook/nllb-200-distilled-600M"
        ),
        WAV2LIP_ASSETS_DIR=str(instance_path / "wav2lip_assets"),
        LIPSYNC_DEFAULT=os.environ.get("LIPSYNC_DEFAULT", "0") == "1",
        HERO_BACKGROUND=random.choice(background_images),
    )

    db.init_app(app)
    login_manager.init_app(app)

    from .models import User  # noqa: WPS433 - imported for login manager

    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    from .auth import auth_bp
    from .routes import main_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)

    with app.app_context():
        db.create_all()

    return app

