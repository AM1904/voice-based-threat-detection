"""
WATZS Server — Email Notifications
====================================
Sends email alerts for Level 3 (HIGH) threat detections
using Gmail SMTP.

Setup:
    1. Enable 2-Factor Authentication on your Gmail account
    2. Generate an App Password: https://myaccount.google.com/apppasswords
    3. Set environment variables or update config/notification_config.json:
        WATZS_EMAIL_SENDER=your.email@gmail.com
        WATZS_EMAIL_PASSWORD=your-app-password
        WATZS_EMAIL_RECIPIENT=recipient@example.com
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


class EmailNotifier:
    """
    Sends email notifications for Level 3 alerts.

    Loads config from environment variables first, then falls back
    to config/notification_config.json.
    """

    def __init__(self):
        self.enabled = False
        self.sender_email = None
        self.sender_password = None
        self.recipient_email = None

        self._lock = threading.Lock()
        self._sent_count = 0
        self._failed_count = 0

        self._load_config()

    def _load_config(self):
        """Load email config from env vars or config file."""
        # Try environment variables first
        self.sender_email = os.environ.get("WATZS_EMAIL_SENDER")
        self.sender_password = os.environ.get("WATZS_EMAIL_PASSWORD")
        self.recipient_email = os.environ.get("WATZS_EMAIL_RECIPIENT")

        # Fall back to config file
        if not all([self.sender_email, self.sender_password, self.recipient_email]):
            if os.path.exists(CONFIG_PATH):
                try:
                    with open(CONFIG_PATH, "r") as f:
                        config = json.load(f)
                    email_cfg = config.get("email", {})
                    self.sender_email = self.sender_email or email_cfg.get("sender_email")
                    self.sender_password = self.sender_password or email_cfg.get("sender_password")
                    self.recipient_email = self.recipient_email or email_cfg.get("recipient_email")
                except Exception as e:
                    print(f"[EmailNotifier] Warning: Failed to read config: {e}")

        # Check if fully configured
        if all([self.sender_email, self.sender_password, self.recipient_email]):
            self.enabled = True
            print(f"[EmailNotifier] OK - Configured -- "
                  f"sender: {self.sender_email}, "
                  f"recipient: {self.recipient_email}")
        else:
            print("[EmailNotifier] WARNING - Not configured -- email notifications disabled.")
            print("  Set env vars: WATZS_EMAIL_SENDER, WATZS_EMAIL_PASSWORD, WATZS_EMAIL_RECIPIENT")
            print(f"  Or create: {CONFIG_PATH}")

    def send_l3_alert(self, event_dict):
        """
        Send an email notification for a Level 3 alert.
        Runs in a background thread to avoid blocking.

        Args:
            event_dict (dict): The alert event data
        """
        if not self.enabled:
            return

        if event_dict.get("level") != 3:
            return

        thread = threading.Thread(
            target=self._send_email,
            args=(event_dict,),
            daemon=True
        )
        thread.start()

    def _send_email(self, event_dict):
        """Send the actual email (runs in background thread)."""
        keyword = event_dict.get("keyword", "Unknown")
        alert_type = event_dict.get("type", "unknown")
        confidence = event_dict.get("confidence", 0)
        timestamp = event_dict.get("timestamp", "N/A")
        source = event_dict.get("source", "N/A")

        subject = f"🔴 WATZS HIGH ALERT (L3) — {keyword}"

        html_body = f"""
        <html>
        <body style="font-family: 'Segoe UI', Arial, sans-serif; background: #1a1a2e; color: #e1e4e8; padding: 24px;">
            <div style="max-width: 500px; margin: 0 auto; background: #16213e; border-radius: 12px; border: 2px solid #e94560; overflow: hidden;">

                <div style="background: #e94560; padding: 16px 24px; text-align: center;">
                    <h1 style="margin: 0; font-size: 20px; color: white;">🔴 WATZS — LEVEL 3 HIGH ALERT</h1>
                </div>

                <div style="padding: 24px;">
                    <table style="width: 100%; border-collapse: collapse; color: #e1e4e8;">
                        <tr>
                            <td style="padding: 8px 0; font-weight: bold; color: #8b949e;">Threat</td>
                            <td style="padding: 8px 0; font-size: 18px; font-weight: 700;">{keyword}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px 0; font-weight: bold; color: #8b949e;">Type</td>
                            <td style="padding: 8px 0;">{alert_type}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px 0; font-weight: bold; color: #8b949e;">Confidence</td>
                            <td style="padding: 8px 0;">{confidence:.0%}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px 0; font-weight: bold; color: #8b949e;">Source</td>
                            <td style="padding: 8px 0;">{source}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px 0; font-weight: bold; color: #8b949e;">Time</td>
                            <td style="padding: 8px 0;">{timestamp}</td>
                        </tr>
                    </table>

                    <div style="margin-top: 20px; padding: 12px; background: #0f3460; border-radius: 8px; text-align: center; color: #f0f3f6;">
                        ⚠️ Immediate attention required. Check the WATZS dashboard.
                    </div>
                </div>

            </div>

            <p style="text-align: center; color: #484f58; font-size: 12px; margin-top: 16px;">
                WATZS — Voice-Based Threat Detection System
            </p>
        </body>
        </html>
        """

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self.sender_email
        msg["To"] = self.recipient_email

        # Plain text fallback
        plain_text = (
            f"WATZS LEVEL 3 HIGH ALERT\n\n"
            f"Threat: {keyword}\n"
            f"Type: {alert_type}\n"
            f"Confidence: {confidence:.0%}\n"
            f"Source: {source}\n"
            f"Time: {timestamp}\n\n"
            f"Immediate attention required."
        )

        msg.attach(MIMEText(plain_text, "plain"))
        msg.attach(MIMEText(html_body, "html"))

        try:
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)

            with self._lock:
                self._sent_count += 1
            print(f"[EmailNotifier] OK - L3 email sent to {self.recipient_email}")

        except smtplib.SMTPAuthenticationError:
            with self._lock:
                self._failed_count += 1
            print("[EmailNotifier] ERROR - Authentication failed!")
            print("  Check your Gmail App Password.")

        except Exception as e:
            with self._lock:
                self._failed_count += 1
            print(f"[EmailNotifier] ERROR - Failed to send email: {e}")

    def update_config(self, sender_email=None, sender_password=None, recipient_email=None):
        """Update email configuration at runtime."""
        if sender_email:
            self.sender_email = sender_email
        if sender_password:
            self.sender_password = sender_password
        if recipient_email:
            self.recipient_email = recipient_email

        self.enabled = all([self.sender_email, self.sender_password, self.recipient_email])
        return self.enabled

    def get_status(self):
        """Get notification system status."""
        return {
            "enabled": self.enabled,
            "sender_email": self.sender_email if self.enabled else None,
            "recipient_email": self.recipient_email if self.enabled else None,
            "sent_count": self._sent_count,
            "failed_count": self._failed_count,
        }


# Singleton instance
notifier = EmailNotifier()
