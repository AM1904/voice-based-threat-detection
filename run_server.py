"""
WATZS — Run Server
===================
Entry point to start the Flask + SocketIO server.

Usage:
    cd repo
    python run_server.py
"""

from server.app import create_app, socketio

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
