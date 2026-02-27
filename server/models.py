"""
WATZS Server — Data Models
===========================
SQLAlchemy models for the threat detection event system.
"""

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timezone, timedelta

db = SQLAlchemy()

IST = timezone(timedelta(hours=5, minutes=30))


class Event(db.Model):
    """Represents a single threat detection event."""

    __tablename__ = "events"

    id = db.Column(db.Integer, primary_key=True)
    event_id = db.Column(db.String(36), unique=True, nullable=False)
    type = db.Column(db.String(50), nullable=False)
    keyword = db.Column(db.String(100))
    level = db.Column(db.Integer, default=1, nullable=False)
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(IST))
    confidence = db.Column(db.Float, default=0.0)
    source = db.Column(db.String(50))

    def to_dict(self):
        return {
            "id": self.id,
            "event_id": self.event_id,
            "type": self.type,
            "keyword": self.keyword,
            "level": self.level,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "confidence": self.confidence,
            "source": self.source,
        }
