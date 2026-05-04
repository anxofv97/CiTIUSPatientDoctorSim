from typing import Tuple, List, Optional
import logging

from .llm import LLM
from .prompts import DOCTOR_STOPPING, DOCTOR_INSIST

logger = logging.getLogger(__name__)

class Doctor:
    """
    Class to simulate a Doctor in a therapy session.

    """
    def __init__(self, model_name: str = "qwen3", max_new_tokens: int = 1024, temperature: float = 0.7, stopping_detection: bool = True):
        self.model = LLM(model_name=model_name)
        if stopping_detection:
            self.stopping_model = LLM(model_name="qwen3")
        else:
            self.stopping_model = None

        # Default generation parameters for the doctor's responses
        self.default_max_new_tokens = max_new_tokens
        self.default_temperature = temperature
        self.stopping_detection = stopping_detection

        self.times_ended = 0  # how many times the doctor has detected the conversation end
        
    def reset_conversation(self, instructions: Optional[str] = None, doctor_greeting: Optional[str] = None):
        """
        Reset/set the conversation state.

        Args:
            instructions: system instructions text (optional)
        """
        self.model.reset_conversation(instructions=instructions)
        self.times_ended = 0
        # Add initial doctor greeting
        if doctor_greeting:
            self.model._history.append({"role": "assistant", "content": doctor_greeting})
    
    def is_conversation_ended(self) -> bool:
        """
        Checks if the conversation should be considered ended based on the doctor's stopping criteria.
        """
        if self.times_ended >= 2: # If detected END two times in a row, consider conversation ended
            return True
        else:
            return False
    
    def _check_conversation_ending(self, last_messages=6) -> bool:
        """
        Checks if the conversation should be considered ended based on the doctor's stopping criteria.

        last_messages: how many of the most recent messages to include in the prompt for the stopping criteria evaluation (default: 6)
        """
        stopping_prompt = "Here is the conversation you have to judge whether it has ended or not."
        recent_messages = self.get_conversation_history()[-last_messages:]
        # Parse conversation history into a prompt for the stopping criteria evaluation. Only include user and assistant messages, ignore system instructions. Format them as "Patient: ..." and "Doctor: ..." to make it clear for the model.
        for msg in recent_messages:
            if msg["role"] != "system":
                if msg["role"] == "user":
                    stopping_prompt += f"Patient: {msg['content'].strip()}\n"
                if msg["role"] == "assistant":
                    stopping_prompt += f"Doctor: {msg['content'].strip()}\n"
        _, stopping_reply = self.stopping_model.run_prompt(stopping_prompt, instructions=DOCTOR_STOPPING, max_new_tokens=32, do_sample=False, keep_history=False)
        if "Yes" in stopping_reply or "yes" in stopping_reply:
            return True
        return False
        
    def get_response(
        self,
        prompt: str,
        instructions: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Sends a prompt to the model.

        Args:
            prompt: user prompt
            instructions: system instructions (optional) — if passed and there is no persistent history, they will be used

        Returns:
            (raw_full_response, final_message_text)
        """
        if self.stopping_detection:
            if self._check_conversation_ending():
                self.times_ended += 1
                logger.info(f"Doctor has detected end of conversation {self.times_ended} times.")
            else:   # Reset the counter if conversation restarts
                self.times_ended = 0
        
        _, doc_reply = self.model.run_prompt(prompt, instructions=instructions, max_new_tokens=self.default_max_new_tokens, temperature=self.default_temperature, keep_history=True)
        # First time doctor detects end of conversation, add the insist prompt to try to continue the conversation
        if self.times_ended == 1:
            doc_reply = doc_reply + "\n" + DOCTOR_INSIST
            self.model._history[-1]["content"] = doc_reply # Update the last assistant message in the history with the insist prompt added
        return _, doc_reply
        
    def get_conversation_history(self) -> List[dict]:
        """Returns the current conversation history as a list of messages (dicts with 'role' and 'content')"""
        return self.model._history
        