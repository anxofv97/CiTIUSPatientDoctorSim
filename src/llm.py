from typing import Tuple, List, Optional
import torch
import logging
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

class LLM:
    """
    Simple class to simulate a LLM.

    Supports one model (selected by `model_name`):
      - "qwen3": uses Qwen/Qwen3-4B-Instruct-2507

    """
    def __init__(self, model_name: str = "qwen3", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None

        # Conversation history
        self._history: List[dict] = []
        self._initialize_model()

    def _initialize_model(self):
        """Load tokenizer and model into memory"""
        if self.model_name == "qwen3":
            model_name_hf = "Qwen/Qwen3-4B-Instruct-2507"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_hf)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_hf,
                torch_dtype=torch.float16,
                device_map="auto",
                offload_folder="offload_dir",
                offload_state_dict=True
            )
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")
        
    def reset_conversation(self, instructions: Optional[str] = None):
        """
        Reset/set the conversation state.

        Args:
            instructions: system instructions text (optional)
        """
        if self.model_name == "qwen3":
            self._history = []
            # add system instructions if any
            if instructions:
                self._history.append({"role": "system", "content": instructions})
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")
        
    def _run_qwen3(self, messages, temperature=0.7, max_new_tokens=1024) -> Tuple[str, str]:
        """
        Run a Qwen3 generation with the given messages as input.

        Args:
            messages: list of dicts with 'role' and 'content' keys, representing the conversation history to use as input
            temperature: sampling temperature
            max_new_tokens: maximum number of tokens to generate
        """
        start_time = datetime.now()

        chat_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        logger.debug(f"Chat prompt: {chat_prompt}")

        inputs = self.tokenizer(chat_prompt, return_tensors="pt").to(self.model.device)

        # Count input tokens for qwen3
        try:
            input_tokens = inputs["input_ids"].shape[1]
        except Exception:
            input_tokens = None

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9
        )

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        # Log: number of input tokens and generated tokens (qwen3)
        try:
            total_tokens = outputs.shape[1] if outputs.dim() == 2 else outputs.shape[0]
            if input_tokens is not None:
                generated_tokens = max(0, total_tokens - input_tokens)
                logger.debug(f"[qwen3] inference completed in {processing_time:.2f} s. Input tokens={input_tokens}, Generated tokens={generated_tokens}")
        except Exception:
            logger.debug("Could not compute token counts for qwen3 response.", exc_info=True)

        start_idx = inputs["input_ids"].shape[1]
        generated_text = self.tokenizer.decode(outputs[0][start_idx:], skip_special_tokens=True)
        logger.debug(f"Generated text: {generated_text}")
        return chat_prompt + generated_text, generated_text
        
    def run_prompt(
        self,
        prompt: str,
        instructions: Optional[str] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        keep_history: bool = False
    ) -> Tuple[str, str]:
        """
        Sends a prompt to the model.

        Args:
            prompt: user prompt
            instructions: system instructions (optional) — if passed and there is no persistent history, they will be used
            max_new_tokens: maximum generation length
            temperature: sampling temperature
            keep_history: if True, the response is added to the internal history

        Returns:
            (raw_full_response, final_message_text)
        """
        # Load model if not already loaded
        if self.model is None or self.tokenizer is None:
            self._initialize_model()

        if self.model_name == "qwen3":  # qwen3
            # If keep_history and no history, initialize
            if keep_history and len(self._history) == 0:
                self.reset_conversation(instructions)
            # Prepare messages (ephemeral or persistent)
            if keep_history:
                # do not mutate history if only a temporary override is desired:
                if instructions:
                    # build a temporary copy of the history with overridden system instructions
                    non_system = [m for m in self._history if m.get("role") != "system"]
                    temp_messages = [{"role": "system", "content": instructions}] + non_system
                    temp_messages.append({"role": "user", "content": prompt})
                    messages = temp_messages
                    self._history.append({"role": "user", "content": prompt})
                else:
                    # normal behavior: use history as is
                    self._history.append({"role": "user", "content": prompt})
                    messages = list(self._history)
            else:
                msgs = []
                if instructions:
                    msgs.append({"role": "system", "content": instructions})
                msgs.append({"role": "user", "content": prompt})
                messages = msgs

            # Run prompt
            full_text, generated_text = self._run_qwen3(messages, temperature=temperature, max_new_tokens=max_new_tokens)

            if keep_history:
                self._history.append({"role": "assistant", "content": generated_text})
            
            return full_text, generated_text
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")
        
    def get_conversation_history(self) -> List[dict]:
        """Returns the current chat history as a list of messages (List[Dict] with 'role' and 'content')"""
        if self.model_name == "qwen3":
            return list(self._history)
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")
        