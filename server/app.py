"""
WATZS Server — Flask Application
==================================
Main application entry point. Flask + Flask-SocketIO + SQLite.

Usage:
    cd repo
    python -m server.app

The server will start on http://localhost:5000
"""

import os
from flask import Flask, send_from_directory
from flask_socketio import SocketIO
from server.models import db
from server.routes import api

socketio = SocketIO()


def create_app():
    """Application factory."""
    app = Flask(__name__)

    # Config
    instance_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "instance")
    os.makedirs(instance_dir, exist_ok=True)

    app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{os.path.join(instance_dir, 'events.db')}"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "watzs-dev-key-change-in-production")

    # Initialize extensions
    db.init_app(app)
    socketio.init_app(app, cors_allowed_origins="*")

    # Register routes
    app.register_blueprint(api)

    # Serve frontend
    frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")

    @app.route("/")
    def serve_dashboard():
        return send_from_directory(frontend_dir, "index.html")

    @app.route("/static/<path:filename>")
    def serve_static(filename):
        return send_from_directory(os.path.join(frontend_dir, "static"), filename)

    # Create tables
    with app.app_context():
        db.create_all()

    return app


# ─── SocketIO Event Handlers ─────────────────────────────────────────────
@socketio.on("connect")
def handle_connect():
    print("[Server] Client connected")


@socketio.on("disconnect")
def handle_disconnect():
    print("[Server] Client disconnected")


# ─── Entry Point ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = create_app()

    print("=" * 50)
    print("  WATZS — Threat Detection Server")
    print("=" * 50)
    print()
    print("  Dashboard:  http://localhost:5000")
    print("  POST alert: http://localhost:5000/alert")
    print("  GET events: http://localhost:5000/events")
    print()

    socketio.run(app, host="0.0.0.0", port=5000, debug=True, allow_unsafe_werkzeug=True)
