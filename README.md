# WATZS — Voice-Based Threat & Emergency Detection System

> Real-time voice and sound monitoring system that detects threats, abnormal sounds, and emergency voice codes using AI.

**Team:** Ananya & Guru | **Duration:** 4 Weeks | **Type:** Internship Prototype

---

## 🏗️ Architecture

```
🎙️ Mic → [Audio Engine]
              ├── Vosk keyword detection (offline STT)
              ├── YAMNet sound classification (scream, gunshot, crash)
              └── Secret voice code counter
                    ↓
           [Alert Classifier]
             → maps to L1 / L2 / L3
                    ↓
           [Flask API] (POST /alert)
             → writes to SQLite
             → emits SocketIO event
             → fires SMS/email on L3
                    ↓
           [Dashboard UI]
             → live level indicator
             → event log table
             → system ON/OFF toggle
```

## 🚨 Alarm Levels

| Level | Trigger | Response | Notification |
|-------|---------|----------|--------------|
| **L1** Low | Single threat keyword | 🟡 Yellow indicator, log entry | None |
| **L2** Medium | Abnormal sound OR multiple keywords OR any word repeated 3+ times | 🟠 Orange indicator, alert | None |
| **L3** High | Secret code 3×, or L1+L2 within 30s, or keyword repeated >2 times | 🔴 Red siren, full-screen overlay | SMS / Email |

## 📁 Project Structure

```
watzs_voice/
├── audio_engine/          ← Ananya
│   ├── __init__.py
│   ├── capture.py          # Mic audio capture
│   ├── keyword_detector.py # Vosk keyword detection
│   ├── sound_classifier.py # YAMNet sound classification
│   ├── voice_code.py       # Secret voice code tracker
│   └── alert_classifier.py # Alert level classifier
├── server/                ← Guru
│   ├── app.py              # Flask application
│   ├── models.py           # SQLAlchemy models
│   ├── routes.py           # API endpoints
│   └── notifier.py         # SMS/Email notifications
├── frontend/
│   ├── index.html          # Dashboard page
│   └── dashboard.js        # Real-time SocketIO client
├── config/
│   ├── keywords.json       # Threat keywords by level
│   └── alert_event_schema.json  # Shared alert event format
├── models/                 # Vosk model files
├── tests/
│   └── test_capture.py     # Mic capture verification
├── requirements.txt
└── README.md
```

## 👥 Team Split

| Member | Module | Responsibility |
|--------|--------|----------------|
| **Ananya** | `audio_engine/` | Audio capture, keyword detection (Vosk), sound classification (YAMNet), voice code logic, alert classifier |
| **Guru** | `server/` + `frontend/` | Flask API, SocketIO, SQLite DB, dashboard UI, notifications |

## 🛠️ Tech Stack

| Layer | Tool | Why |
|-------|------|-----|
| Audio capture | PyAudio + SoundDevice | Standard Python mic streaming |
| Speech/keywords | Vosk (offline) | Works without internet; has Indian English model |
| Sound classification | YAMNet (TensorFlow Hub) | Pre-trained for 521 sound classes |
| Backend/API | Flask | Lightweight Python framework |
| Real-time comms | Flask-SocketIO | Live push to dashboard |
| Frontend | HTML + Tailwind + vanilla JS | Simple, no framework overhead |
| Database | SQLite + SQLAlchemy | Zero setup, local |
| Notifications | Twilio SMS / Gmail SMTP | Free tier sufficient |

## 🚀 Setup

### Prerequisites
- Python 3.9+
- Working microphone

### Installation
```bash
# Clone the repo
git clone <repo-url>
cd watzs_voice

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Download Vosk model (Indian English)
# Model will be placed in models/ directory
```

### Running
```bash
# Test mic capture
python tests/test_capture.py

# Start audio engine (Phase 2+)
python -m audio_engine.capture
```

---

*WATZS Prototype · Ananya & Guru · Internship Project 2026*
