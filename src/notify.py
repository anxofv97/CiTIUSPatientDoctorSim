import os
import requests
from datetime import datetime
import time
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)

# THESE ENV VARS MUST BE SET TO ENABLE TELEGRAM NOTIFICATIONS. Create Bot with BotFather and get your token and chat_id to receive private notifications.
TOKEN = os.getenv("TG_TOKEN")
CHAT_ID = os.getenv("TG_CHAT_ID")

# Save in-memory process starts: process_name -> (timestamp_unix_start, formatted_start_str)
_starts: Dict[str, Tuple[float, str]] = {}


def _now_local_str() -> str:
    """Returns the local date/time formatted for the log prefix."""
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")


def send_telegram(text: str) -> None:
    """
    Sends a message via Telegram (if credentials are available) or prints it.
    Always adds the log prefix (local date/time) to the message.
    """
    log_text = f"{_now_local_str()} - {text}"

    if not TOKEN or not CHAT_ID:
        logger.warning("Telegram credentials not configured. Message not sent.")
        return

    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": log_text}
    try:
        r = requests.post(url, json=data, timeout=10)
        r.raise_for_status()
    except Exception as e:
        # Log the error with prefix, do not propagate
        logger.exception(f"Error sending Telegram: {e}")
    else:
        logger.info(f"Telegram message sent: {log_text}")


def _format_elapsed_hms(seconds: float) -> str:
    """
    Formats seconds into HH:MM:SS (two digits per component).
    """
    total = int(round(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _format_start_ts_local(ts: float) -> str:
    """
    Formats a timestamp (seconds since epoch) to 'YYYY-MM-DD HH:MM:SS' in the local timezone.
    """
    dt_local = datetime.fromtimestamp(ts).astimezone()
    return dt_local.strftime("%Y-%m-%d %H:%M:%S")


def notify_start_process(process_name: str, extra_info=None) -> None:
    """
    Registers the start of `process_name` (in memory) using time.time() and sends a message with a local time timestamp.
    """
    try:
        if not TOKEN or not CHAT_ID:
            logger.warning("Telegram credentials not configured.")
            return
        ts = time.time()
        start_str = _format_start_ts_local(ts)
        _starts[process_name] = (ts, start_str)
        send_telegram(f"'{process_name}' started.")
        if extra_info:
            send_telegram(f"Additional information: {extra_info}")
    except Exception as e:
        logger.exception(f"Error in notify_start_process: {e}")


def notify_end_process(process_name: str, extra_info=None) -> None:
    """
    If a start is registered, calculates the duration using time.time() and sends a summary
    with the duration formatted as HH:MM:SS. If not found, notifies that no start was found.
    """
    try:
        if not TOKEN or not CHAT_ID:
            logger.warning(f"Telegram credentials not configured.")
            return
        entry = _starts.pop(process_name, None)
        if entry is None:
            send_telegram(f"⚠️ No start record found for '{process_name}'.")
            return

        ts_start, start_str = entry
        elapsed = time.time() - ts_start
        elapsed_str = _format_elapsed_hms(elapsed)
        send_telegram(
            f"✅ '{process_name}' has finished. Duration: {elapsed_str}"
        )
        if extra_info:
            send_telegram(f"Additional information: {extra_info}")
    except Exception as e:
        logger.exception(f"Error in notify_end_process: {e}")