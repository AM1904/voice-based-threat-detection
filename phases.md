# WATZS – Project Phases
**Voice-Based Threat & Emergency Detection System**
*Ananya & Guru | Internship Prototype*

---

## Phase 1 — Setup & Foundation
> **Duration:** Week 1 | **Owner:** Both

### Goals
Get the development environment ready, agree on shared contracts, and ensure
both members can start building independently without blockers.

### Tasks

| Task | Owner | Done? |
|------|-------|-------|
| Create GitHub repo with `main`, `dev`, `feature/*` branch structure | Both | ☐ |
| Add README skeleton and folder structure | Both | ☐ |
| Define shared JSON alert event format: `{id, type, keyword, level, timestamp}` | Both | ☐ |
| Define `keywords.json` config schema (threat words + secret code phrase) | Both | ☐ |
| Set up Python virtual environment, install PyAudio + Vosk | Ananya | ☐ |
| Test laptop mic capture — verify audio stream reads correctly | Ananya | ☐ |
| Set up Flask app scaffold (`app.py`, `models.py`, `routes.py`) | Guru | ☐ |
| Create SQLite DB with `events` table | Guru | ☐ |
| Build bare-bones `index.html` dashboard stub | Guru | ☐ |

### Exit Criteria
- Both members can run their modules locally
- JSON event format is documented and agreed upon
- GitHub repo is live with correct branch structure

---

## Phase 2 — Core Detection Engine
> **Duration:** Week 2 | **Owner:** Ananya (primary), Guru (backend)

### Goals
Build and individually test keyword detection, sound classification, and the
secret voice code logic. In parallel, stand up the Flask API and SocketIO connection.

### Tasks

| Task | Owner | Done? |
|------|-------|-------|
| Implement Vosk live keyword detection on audio stream | Ananya | ☐ |
| Load `keywords.json` and scan transcribed speech for threat words | Ananya | ☐ |
| Integrate YAMNet model for abnormal sound classification | Ananya | ☐ |
| Test YAMNet with sample audio files (scream, gunshot, crash) | Ananya | ☐ |
| Build secret voice code tracker — count phrase in sliding time window | Ananya | ☐ |
| Build `POST /alert` and `GET /events` Flask REST endpoints | Guru | ☐ |
| Integrate Flask-SocketIO, emit `alert_event` on each alert | Guru | ☐ |
| Wire frontend JS to SocketIO — live-update indicator and event log | Guru | ☐ |
| Test each module independently using mock/sample data | Both | ☐ |

### Exit Criteria
- Speaking a keyword into mic fires a detection event (printed to console)
- POSTing a mock alert to Flask updates the dashboard live
- Both modules tested independently before integration

---

## Phase 3 — Alerts, Alarms & Integration
> **Duration:** Week 3 | **Owner:** Both

### Goals
Connect both modules into one running system, implement full alarm level logic,
build the complete dashboard UI, and add notification dispatch.

### Tasks

| Task | Owner | Done? |
|------|-------|-------|
| Build alarm level classifier (L1/L2/L3 with escalation rules) | Ananya | ☐ |
| Wire audio engine output → `POST /alert` Flask endpoint | Ananya | ☐ |
| Build complete dashboard UI — colour-coded alarm cards, event log, toggle | Guru | ☐ |
| Implement Level 3 notification — Gmail SMTP (primary) or Twilio SMS | Guru | ☐ |
| Full end-to-end integration test — mic → engine → API → SocketIO → UI | Both | ☐ |
| Cross-review each other's code via GitHub PRs | Both | ☐ |
| Merge all feature branches into `dev`, resolve any conflicts | Both | ☐ |

### Alarm Level Reference

| Level | Trigger | Response | Notification |
|-------|---------|----------|--------------|
| **L1** Low | Single threat keyword | 🟡 Yellow indicator, log entry | None |
| **L2** Medium | Abnormal sound OR multiple keywords | 🟠 Orange indicator, alert | None |
| **L3** High | Secret code 3×, or L1+L2 within 30s | 🔴 Red siren, full-screen overlay | SMS / Email |

### Exit Criteria
- Keyword → L1 on dashboard ✓
- Scream audio → L2 ✓
- Secret code 3× → L3 + notification fires ✓
- All events logged to SQLite correctly ✓

---

## Phase 4 — Testing, Polish & Demo
> **Duration:** Week 4 | **Owner:** Both

### Goals
Harden the system, reduce false positives, polish the UI, build Demo Mode,
and prepare everything for the internship presentation.

### Tasks

| Task | Owner | Done? |
|------|-------|-------|
| End-to-end scenario testing — keyword, sound, and secret code paths | Both | ☐ |
| Tune YAMNet confidence threshold to reduce false positives | Ananya | ☐ |
| Test Vosk with real voices — switch to Indian English model if needed | Ananya | ☐ |
| Test edge cases: background music, normal conversation, simultaneous sounds | Ananya | ☐ |
| Build **Demo Mode** — simulate L1/L2/L3 on button click, no mic needed | Guru | ☐ |
| Polish dashboard UI — clean layout, status indicators, reset button | Guru | ☐ |
| Confirm email/SMS fires correctly end-to-end | Guru | ☐ |
| Write full README — setup guide, how to run, architecture, feature list | Both | ☐ |
| Record demo video / screen recording (submission backup) | Both | ☐ |
| Final merge `dev` → `main`, tag release `v1.0` | Both | ☐ |
| Prepare live demo script for presentation day | Both | ☐ |

### Demo Script Outline
1. Start system — show dashboard in "monitoring" state
2. Speak a threat keyword → L1 triggers live
3. Play scream audio file → L2 triggers
4. Speak secret code phrase 3× → L3 triggers, notification fires
5. Show event log — all entries timestamped and categorised
6. Switch to Demo Mode → repeat all levels cleanly with buttons

### Exit Criteria
- All three alarm levels work reliably end-to-end
- Demo Mode works without mic
- README is complete and repo is presentable
- `main` is clean, tagged `v1.0`, ready to share

---

## Summary
```
Week 1 ── Setup & Foundation      ── Dev environment, repo, shared contracts
Week 2 ── Core Detection Engine   ── Keyword detection, sound classifier, voice code
Week 3 ── Alerts & Integration    ── Alarm levels, full UI, notifications, wire-up
Week 4 ── Testing & Demo          ── Hardening, polish, Demo Mode, presentation
```

---
*WATZS Prototype · Ananya & Guru · Internship Project 2026*