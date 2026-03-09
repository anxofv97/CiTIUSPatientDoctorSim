from typing import Tuple, List, Optional
import logging
import requests
from datetime import datetime

logger = logging.getLogger(__name__)

class Patient:
    """
    Class to simulate a Patient in a therapy session.

    """
    def __init__(self, patient_url: str = "http://127.0.0.1:6416/v1/chat/completions"):
        self.patient_url = patient_url
        self.patient_messages = None
    
    def get_response(self, doctor_message: str) -> Tuple[str, str]:
        """
        Generate a patient response to the doctor's message.

        Args:
            doctor_message: the latest message from the doctor to respond to.

        Returns:
            pat_msg: the patient's response message content.
        """
        if self.patient_messages is None:
            raise ValueError("Conversation not initialized. Please call reset_conversation() first.")
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer sk-local"
            }
            # Append doctor's latest message to patient conversation history
            self.patient_messages.append({"role": "user", "content": doctor_message})
            payload = {"messages": self.patient_messages}
            resp = requests.post(self.patient_url, headers=headers, json=payload)
            resp.raise_for_status()
            resp_json = resp.json()
            pat_msg = resp_json["choices"][0]["message"]["content"].strip()
            # Append patient assistant reply to patient history
            self.patient_messages.append({"role": "assistant", "content": pat_msg})
        except Exception as e:
            logger.exception("Patient request failed")
        return pat_msg

    def reset_conversation(self, instructions: Optional[str] = None):
        """
        Reset/set the conversation state.

        Args:
            instructions: system instructions text (optional)
        """
        if instructions:
            self.patient_messages = [{"role": "system", "content": instructions}]
        else:
            self.patient_messages = []

    def get_conversation_history(self) -> List[dict]:
        """
        Get the current conversation history as a list of messages.

        Returns:
            A list of message dicts with "role" and "content" keys.
        """
        if self.patient_messages is None:
            raise ValueError("Conversation not initialized. Please call reset_conversation() first.")
        return self.patient_messages