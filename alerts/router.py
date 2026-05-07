import os
import smtplib
from email.mime.text import MIMEText

import requests
import yaml

_SEVERITY_ORDER = {"INFO": 0, "WARNING": 1, "CRITICAL": 2}


class AlertRouter:
    def __init__(self, config_path: str = "configs/alert_routing.yaml"):
        with open(config_path) as f:
            self._config = yaml.safe_load(f)

    def dispatch(self, alert) -> None:
        channels = self._config.get("channels", {})
        if channels.get("slack", {}).get("enabled"):
            self._send_slack(alert, channels["slack"])
        if channels.get("email", {}).get("enabled"):
            self._send_email(alert, channels["email"])
        if channels.get("webhook", {}).get("enabled"):
            self._send_webhook(alert, channels["webhook"])

    def _meets_severity(self, actual: str, minimum: str) -> bool:
        return _SEVERITY_ORDER.get(actual, 0) >= _SEVERITY_ORDER.get(minimum, 0)

    def _send_slack(self, alert, config: dict) -> None:
        if not self._meets_severity(alert.severity, config.get("min_severity", "WARNING")):
            return
        url = os.path.expandvars(config["webhook_url"])
        try:
            requests.post(url, json={
                "text": f"*[{alert.severity}]* {alert.title}\n{alert.message}",
                "username": "ML Observability",
            }, timeout=5)
        except requests.RequestException:
            pass

    def _send_email(self, alert, config: dict) -> None:
        if not self._meets_severity(alert.severity, config.get("min_severity", "CRITICAL")):
            return
        try:
            msg = MIMEText(f"{alert.message}\n\n{alert.explanation or ''}")
            msg["Subject"] = f"[{alert.severity}] {alert.title}"
            msg["From"] = config["from"]
            msg["To"] = ", ".join(config["to"])
            with smtplib.SMTP(os.path.expandvars(config["smtp_host"]), config["smtp_port"]) as smtp:
                smtp.starttls()
                smtp.send_message(msg)
        except Exception:
            pass

    def _send_webhook(self, alert, config: dict) -> None:
        if not self._meets_severity(alert.severity, config.get("min_severity", "WARNING")):
            return
        try:
            requests.post(os.path.expandvars(config["url"]), json={
                "severity": alert.severity,
                "title": alert.title,
                "message": alert.message,
                "model_id": alert.model_id,
            }, timeout=5)
        except requests.RequestException:
            pass
