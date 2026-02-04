# Voice-Based Threat & Emergency Detection System

##  Project Overview
The **Voice-Based Threat & Emergency Detection System** is an intelligent security solution designed to detect emergency and threat situations in real time using **audio analysis and speech recognition**.  
The system continuously listens to audio from microphones or IP cameras, analyzes speech and sound patterns, and triggers alerts or alarms based on detected threats.

This project aims to improve safety by enabling **hands-free, silent, and automated emergency detection**, especially in high-risk environments.

---

##  Objectives
- Continuous audio monitoring
- Detection of threat-related keywords
- Identification of abnormal sounds (panic, shouting, screams)
- Secret emergency activation using voice codes
- Multi-level alert generation
- Alarm and notification triggering
- Real-time logging and monitoring

---

## System Architecture (High-Level)
The system follows a **layered and modular architecture**:

1. **Input Layer**
   - Microphone / IP Camera audio capture

2. **Analysis Layer**
   - Speech-to-text processing
   - Keyword detection
   - Sound intensity (decibel) analysis

3. **Decision Layer**
   - Alert level classification
   - Emergency voice code recognition
   - Threat severity evaluation

4. **Output Layer**
   - Dashboard alerts
   - Alarm siren activation
   - Optional SMS / notifications

This modular structure allows independent development and easy integration.

---

## Features
- Real-time audio processing
- Keyword-based threat detection
- Abnormal sound recognition
- Discreet emergency voice codes
- Multi-level alert system
- Scalable and extensible design

---

## 🛠️ Technology Stack

### Hardware
- IP Camera with Microphone / External Microphone
- Processing Unit (PC / Raspberry Pi / ESP32)
- Alarm Siren
- Network Router
- Power Supply
- Optional GSM Module (SMS alerts)

### Software
- Programming Language: **Python**
- Speech-to-Text Engine (e.g., Whisper / Vosk)
- Audio Processing Libraries
- Database: SQLite / MySQL
- Dashboard: Web or Desktop Application
- Version Control: **Git & GitHub**

---

##  Threat Keywords & Emergency Voice Codes

### Sample Threat Keywords
- help
- emergency
- robbery
- gun
- save me
- don’t move

### Emergency Voice Codes (Examples)
- code red
- code blue
- alpha alert

Voice codes are repeated discreet phrases that trigger high-level alerts without alerting attackers.

---

##  Alert Levels

| Level | Description |
|-----|-------------|
| Level 1 | Low alert – suspicious keywords |
| Level 2 | Medium alert – panic sounds or repeated keywords |
| Level 3 | High alert – emergency voice codes or severe threats |

---

## 👥 Team & Collaboration

This project is developed by a **team of two members**, following a parallel development strategy.

### Role Distribution
- **Member 1**: Audio capture, speech recognition, sound analysis
- **Member 2**: Alert logic, emergency rules, dashboard & notifications


