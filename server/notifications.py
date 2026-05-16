"""
WATZS Server — Notifications (Email & SMS)
===========================================
Sends alerts for Level 3 (HIGH) threat detections.
- Email: via Gmail SMTP
- SMS: via Twilio API

Setup:
    1. Update config/notification_config.json with your credentials.
    2. For Email: Enable 2FA on Gmail and create an App Password.
    3. For SMS: Get Account SID, Auth Token, and a Twilio phone number.
"""

import os
import json
import smtplib
import threading
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ─── Default Configuration ──────────────────────────────────────────────
CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config", "notification_config.json"
)

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587


class AlertNotifier:
    """
    Handles both Email and SMS notifications for Level 3 alerts.
    """

    def __init__(self):
        # Email settings
        self.email_enabled = False
        self.sender_email = None
        self.sender_password = None
        self.recipient_email = None

        # SMS settings
        self.sms_enabled = False
        self.twilio_sid = None
        self.twilio_auth_token = None
        self.twilio_from_number = None
        self.recipient_phone = None

        self._lock = threading.Lock()
        self._sent_emails = 0
        self._sent_sms = 0
        self._failed_count = 0

        self._load_config()

    def _load_config(self):
        """Load configuration from env vars or config file."""
        # Fallback to config file
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, "r") as f:
                    config = json.load(f)
                
                # Email Config
                e_cfg = config.get("email", {})
                self.sender_email = os.environ.get("WATZS_EMAIL_SENDER") or e_cfg.get("sender_email")
                self.sender_password = os.environ.get("WATZS_EMAIL_PASSWORD") or e_cfg.get("sender_password")
                self.recipient_email = os.environ.get("WATZS_EMAIL_RECIPIENT") or e_cfg.get("recipient_email")

                # SMS Config
                s_cfg = config.get("sms", {})
                self.twilio_sid = os.environ.get("TWILIO_ACCOUNT_SID") or s_cfg.get("account_sid")
                self.twilio_auth_token = os.environ.get("TWILIO_AUTH_TOKEN") or s_cfg.get("auth_token")
                self.twilio_from_number = os.environ.get("TWILIO_FROM_NUMBER") or s_cfg.get("from_number")
                self.recipient_phone = os.environ.get("WATZS_RECIPIENT_PHONE") or s_cfg.get("recipient_phone")

            except Exception as e:
                print(f"[Notifier] Warning: Failed to read config: {e}")

        # Check Email enablement
        if all([self.sender_email, self.sender_password, self.recipient_email]):
            self.email_enabled = True
            print(f"[Notifier] Email OK: {self.sender_email} -> {self.recipient_email}")
        
        # Check SMS enablement
        if all([self.twilio_sid, self.twilio_auth_token, self.twilio_from_number, self.recipient_phone]):
            self.sms_enabled = True
            print(f"[Notifier] SMS OK: {self.twilio_from_number} -> {self.recipient_phone}")
        else:
            print("[Notifier] SMS WARNING: Twilio credentials missing in config.")

    def send_l3_alert(self, event_dict):
        """Send all enabled notifications for an L3 alert."""
        if event_dict.get("level") != 3:
            return

        # Send Email
        if self.email_enabled:
            threading.Thread(target=self._send_email, args=(event_dict,), daemon=True).start()

        # Send SMS
        if self.sms_enabled:
            threading.Thread(target=self._send_sms, args=(event_dict,), daemon=True).start()

    def _send_email(self, event_dict):
        """Internal method to send email."""
        keyword = event_dict.get("keyword", "Unknown")
        alert_type = event_dict.get("type", "unknown")
        confidence = event_dict.get("confidence", 0)
        timestamp = event_dict.get("timestamp", "N/A")

        subject = f"🔴 WATZS HIGH ALERT (L3) — {keyword}"
        
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self.sender_email
        msg["To"] = self.recipient_email

        plain_text = (
            f"WATZS LEVEL 3 HIGH ALERT\n\n"
            f"Threat: {keyword}\n"
            f"Type: {alert_type}\n"
            f"Confidence: {confidence:.0%}\n"
            f"Time: {timestamp}\n\n"
            f"Immediate attention required."
        )
        msg.attach(MIMEText(plain_text, "plain"))

        try:
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            with self._lock: self._sent_emails += 1
            print(f"[Notifier] Email sent to {self.recipient_email}")
        except Exception as e:
            with self._lock: self._failed_count += 1
            print(f"[Notifier] Email Error: {e}")

    def _send_sms(self, event_dict):
        """Internal method to send SMS via Twilio."""
        try:
            from twilio.rest import Client
            
            keyword = event_dict.get("keyword", "Unknown")
            level = event_dict.get("level", 3)
            
            client = Client(self.twilio_sid, self.twilio_auth_token)
            
            message_body = (
                f"WATZS Notification: Threat detected ({keyword}). "
                f"Please check your system dashboard."
            )
            
            message = client.messages.create(
                body=message_body,
                from_=self.twilio_from_number,
                to=self.recipient_phone
            )
            
            with self._lock: self._sent_sms += 1
            print(f"[Notifier] SMS sent to {self.recipient_phone} (SID: {message.sid})")
            
        except ImportError:
            print("[Notifier] SMS Error: 'twilio' library not installed.")
        except Exception as e:
            with self._lock: self._failed_count += 1
            print(f"[Notifier] SMS Error: {e}")

    def get_status(self):
        """Get status of all notification channels."""
        return {
            "email": {
                "enabled": self.email_enabled,
                "sent": self._sent_emails,
                "recipient": self.recipient_email
            },
            "sms": {
                "enabled": self.sms_enabled,
                "sent": self._sent_sms,
                "recipient": self.recipient_phone
            },
            "total_failed": self._failed_count
        }


# Singleton instance
notifier = AlertNotifier()
