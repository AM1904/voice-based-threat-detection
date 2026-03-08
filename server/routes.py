"""
WATZS Server — API Routes
==========================
REST endpoints for the threat detection system.

Endpoints:
    POST /alert                — Receive and store an alert from the audio engine
    GET  /events               — List all events (optional ?limit=N)
    GET  /events/latest        — Get the most recent event
    DELETE /events             — Clear all events (dev/testing only)
    GET  /notifications/status — Check email notification status
"""

import uuid
from flask import Blueprint, request, jsonify
from server.models import db, Event
from server.notifications import notifier

api = Blueprint("api", __name__)


@api.route("/alert", methods=["POST"])
def post_alert():
    """
    Receive an alert event from the audio engine.

    Expected JSON body:
        {
            "type": "keyword" | "sound" | "voice_code" | "repetition",
            "keyword": "string",
            "level": 1 | 2 | 3,
            "confidence": 0.0-1.0  (optional),
            "source": "string"     (optional)
        }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    alert_type = data.get("type")
    level = data.get("level")

    if not alert_type:
        return jsonify({"error": "Missing required field: type"}), 400
    if alert_type not in ("keyword", "sound", "voice_code", "repetition"):
        return jsonify({"error": f"Invalid type: {alert_type}"}), 400
    if level is None:
        return jsonify({"error": "Missing required field: level"}), 400
    if level not in (1, 2, 3):
        return jsonify({"error": f"Invalid level: {level}. Must be 1, 2, or 3"}), 400

    event = Event(
        event_id=data.get("id", str(uuid.uuid4())),
        type=alert_type,
        keyword=data.get("keyword"),
        level=level,
        confidence=data.get("confidence", 0.0),
        source=data.get("source"),
    )
    db.session.add(event)
    db.session.commit()

    event_dict = event.to_dict()

    # Emit via SocketIO
    from server.app import socketio
    socketio.emit("alert_event", event_dict)

    # Trigger email notification for L3 alerts
    if level == 3:
        notifier.send_l3_alert(event_dict)

    return jsonify(event_dict), 201


@api.route("/events", methods=["GET"])
def get_events():
    """
    Get all events, ordered by most recent first.
    Optional query param: ?limit=N
    """
    limit = request.args.get("limit", type=int)
    query = Event.query.order_by(Event.id.desc())

    if limit and limit > 0:
        query = query.limit(limit)

    events = query.all()
    return jsonify([e.to_dict() for e in events]), 200


@api.route("/events/latest", methods=["GET"])
def get_latest_event():
    """Get the most recent event."""
    event = Event.query.order_by(Event.id.desc()).first()
    if not event:
        return jsonify({"message": "No events recorded yet"}), 200
    return jsonify(event.to_dict()), 200


@api.route("/events", methods=["DELETE"])
def clear_events():
    """Clear all events. For development/testing only."""
    count = Event.query.delete()
    db.session.commit()
    return jsonify({"message": f"Deleted {count} events"}), 200


@api.route("/notifications/status", methods=["GET"])
def notifications_status():
    """Check the email notification system status."""
    return jsonify(notifier.get_status()), 200


# ─── Demo Mode ─────────────────────────────────────────────────────────
# Simulates L1/L2/L3 alerts via dashboard buttons — no mic needed.

DEMO_SCENARIOS = {
    1: {
        "type": "keyword",
        "keyword": "help",
        "level": 1,
        "confidence": 0.92,
        "source": "demo",
        "metadata": {"description": "Simulated L1 keyword detection"},
    },
    2: {
        "type": "sound",
        "keyword": "scream",
        "level": 2,
        "confidence": 0.78,
        "source": "demo",
        "metadata": {"description": "Simulated L2 threat sound", "sound_class": "scream"},
    },
    3: {
        "type": "voice_code",
        "keyword": "watzs emergency",
        "level": 3,
        "confidence": 0.95,
        "source": "demo",
        "metadata": {"description": "Simulated L3 secret code activation"},
    },
}


@api.route("/demo/trigger", methods=["POST"])
def demo_trigger():
    """
    Simulate an alert at a given level for Demo Mode.

    JSON body:
        { "level": 1 | 2 | 3 }

    Optionally override:
        { "level": 2, "keyword": "gunshot", "type": "sound" }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    level = data.get("level")
    if level not in (1, 2, 3):
        return jsonify({"error": "level must be 1, 2, or 3"}), 400

    scenario = dict(DEMO_SCENARIOS[level])

    # Allow overrides from request
    if "keyword" in data:
        scenario["keyword"] = data["keyword"]
    if "type" in data and data["type"] in ("keyword", "sound", "voice_code", "repetition"):
        scenario["type"] = data["type"]

    event = Event(
        event_id=str(uuid.uuid4()),
        type=scenario["type"],
        keyword=scenario["keyword"],
        level=scenario["level"],
        confidence=scenario.get("confidence", 0.9),
        source="demo",
    )
    db.session.add(event)
    db.session.commit()

    event_dict = event.to_dict()

    from server.app import socketio
    socketio.emit("alert_event", event_dict)

    # L3 still triggers notification in demo mode (to test the full path)
    if level == 3:
        notifier.send_l3_alert(event_dict)

    return jsonify(event_dict), 201

