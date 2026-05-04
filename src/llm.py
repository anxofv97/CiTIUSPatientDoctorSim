from typing import Tuple, List, Optional
import os
import re
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
        # Optional rolling window size (None == full history). Try to parse from model_name like '_w5'
        self.window_size: Optional[int] = None
        m = re.search(r"_w(\d+)", str(model_name))
        if m:
            try:
                self.window_size = int(m.group(1))
            except Exception:
                self.window_size = None
        self._initialize_model()

    def _initialize_model(self):
        """Load tokenizer and model into memory"""
        # If model_name points to a local folder, prefer loading from there (e.g., qwen-annomi-therapist_w5)
        if os.path.exists(self.model_name) and os.path.isdir(self.model_name):
            model_dir = self.model_name
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                torch_dtype=torch.float16,
                device_map="auto",
                offload_folder="offload_dir",
                offload_state_dict=True,
            )
            return

        if self.model_name == "qwen3":
            model_name_hf = "Qwen/Qwen3-4B-Instruct-2507"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_hf)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_hf,
                torch_dtype=torch.float16,
                device_map="auto",
                offload_folder="offload_dir",
                offload_state_dict=True,
            )
            return

        # Otherwise try to load by model identifier (HF)
        try:
            model_name_hf = self.model_name
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_hf)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_hf,
                torch_dtype=torch.float16,
                device_map="auto",
                offload_folder="offload_dir",
                offload_state_dict=True,
            )
            return
        except Exception as e:
            raise ValueError(f"Unsupported model name or failed to load: {self.model_name} ({e})")
        
    def reset_conversation(self, instructions: Optional[str] = None):
        """
        Reset/set the conversation state.

        Args:
            instructions: system instructions text (optional)
        """
        # Reset history for any loaded model; add system instructions if provided
        self._history = []
        if instructions:
            self._history.append({"role": "system", "content": instructions})
        
    def _run_qwen3(self, messages, temperature=0.7, do_sample=True, max_new_tokens=256) -> Tuple[str, str]:
        """
        Run a Qwen3 generation with the given messages as input.

        Args:
            messages: list of dicts with 'role' and 'content' keys, representing the conversation history to use as input
            temperature: sampling temperature
            do_sample: whether to use sampling
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
            do_sample=do_sample,
            temperature=temperature,
            top_p=0.9
        )

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        # Logging: number of input tokens and generated tokens (qwen3). Truncation detection: if generated tokens >= max_new_tokens
        try:
            total_tokens = outputs.shape[1] if outputs.dim() == 2 else outputs.shape[0]
            if input_tokens is not None:
                generated_tokens = max(0, total_tokens - input_tokens)
                # Detect truncation: if generated tokens >= max_new_tokens, the model likely hit the token limit
                truncated = generated_tokens >= max_new_tokens
                logger.info(f"[qwen3] inference completed in {processing_time:.2f} s. Input tokens={input_tokens}, Generated tokens={generated_tokens}")
                if truncated:
                    logger.warning(f"[qwen3] response likely truncated: generated_tokens={generated_tokens} >= max_new_tokens={max_new_tokens}")
                    # Log the raw response for debugging truncation issues
                    raw_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    logger.warning(f"[qwen3] raw response (truncated): {raw_response}")
        except Exception:
            logger.warning("Could not compute token counts for qwen3 response.", exc_info=True)

        start_idx = inputs["input_ids"].shape[1]
        generated_text = self.tokenizer.decode(outputs[0][start_idx:], skip_special_tokens=True)
        # Clean tool/thinking annotations that may remain from templates or plugins. Usually appears in fine tuned models
        try:
            # remove full think/tool_call blocks
            generated_text = re.sub(r"<think>.*?</think>", "", generated_text, flags=re.S)
            generated_text = re.sub(r"<tool_call>.*?</tool_call>", "", generated_text, flags=re.S)
            # remove any leftover opening/closing tags like </think>, <think>, </tool_call>, <tool_call>
            generated_text = re.sub(r"</?(think|tool_call)[^>]*>", "", generated_text)
            generated_text = generated_text.strip()
        except Exception:
            pass
        logger.debug(f"Generated text: {generated_text}")
        # Return cleaned output
        return chat_prompt + generated_text, generated_text
        
    def run_prompt(
        self,
        prompt: str,
        instructions: Optional[str] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        do_sample: bool = True,
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

        # Initialize history if requested
        if keep_history and len(self._history) == 0:
            self.reset_conversation(instructions)

        # Prepare messages (ephemeral or persistent)
        if keep_history:
            # When keeping history, apply rolling-window if configured.
            # Build non-system messages list (user/assistant) preserving order
            non_system = [m for m in self._history if m.get("role") != "system"]

            if instructions:
                # override system instructions for this call
                temp_system = {"role": "system", "content": instructions}
            else:
                # keep original system message if present
                sys_msgs = [m for m in self._history if m.get("role") == "system"]
                temp_system = sys_msgs[0] if len(sys_msgs) > 0 else None

            # Append current user prompt to history
            non_system.append({"role": "user", "content": prompt})
            # Update persistent history with the new user message
            self._history.append({"role": "user", "content": prompt})

            # Apply rolling window to non_system messages if window_size is set
            if self.window_size is not None:
                windowed = non_system[-self.window_size:]
            else:
                windowed = non_system

            messages = []
            if temp_system is not None:
                messages.append(temp_system)
            messages.extend(windowed)
        else:
            msgs = []
            if instructions:
                msgs.append({"role": "system", "content": instructions})
            msgs.append({"role": "user", "content": prompt})
            messages = msgs

        # Run prompt
        full_text, generated_text = self._run_qwen3(messages, temperature=temperature, do_sample=do_sample, max_new_tokens=max_new_tokens)
        if keep_history:
            self._history.append({"role": "assistant", "content": generated_text})
        return full_text, generated_text
        
    def get_conversation_history(self) -> List[dict]:
        """Returns the current chat history as a list of messages (List[Dict] with 'role' and 'content')"""
        return list(self._history)
        