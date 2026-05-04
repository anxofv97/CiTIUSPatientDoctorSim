import pandas as pd
import torch
import re
import random
import argparse
import os
import logging
import logging.config
import json
from typing import List, Any
from datasets import Dataset, load_dataset
from unsloth import FastLanguageModel
from transformers import TrainingArguments, EarlyStoppingCallback, TrainerCallback
from trl import SFTTrainer
from transformers import DataCollatorForLanguageModeling

from src.notify import notify_start_process, notify_end_process
from src.prompts import DOCTOR_SFT

try:
    import psutil
except Exception:
    psutil = None

# Config logging from file
def setup_logging(config_file="./logging_config.json"):
    with open(config_file, "r") as f:
        config = json.load(f)
    logging.config.dictConfig(config)
setup_logging()
logger = logging.getLogger(__name__)

############################################
# 1. CONFIG
############################################

MODEL_NAME = "unsloth/Qwen3-4B-Instruct-2507"
DEFAULT_ANNOMI_PATH = "AnnoMI-full.csv"
OUTPUT_DIR = "models/qwen-annomi-therapist"
MAX_SEQ_LENGTH = 2048
SYSTEM_PROMPT = DOCTOR_SFT

############################################
# 2. Dataset loaders / normalizers
############################################

def load_annomi_dataset(path: str) -> pd.DataFrame:
    logger.info(f"Loading AnnoMI CSV from {path}...")
    df = pd.read_csv(path)
    logger.info(f"Loaded dataset '{path}' with {len(df)} rows and {len(df.columns)} columns")

    # If the dataset includes an `mi_quality` column, keep only rows labeled "high"
    if "mi_quality" in df.columns:
        original_len = len(df)
        df = df[df["mi_quality"].astype(str).str.lower() == "high"].reset_index(drop=True)
        logger.info(f"Filtered dataset by mi_quality='high': {len(df)} / {original_len} rows kept")
    else:
        logger.warning("Warning: 'mi_quality' column not found — no filtering applied")

    return df


def load_cactus_rows() -> List[Any]:
    logger.info(f"Loading Hugging Face dataset LangAGI-Lab/cactus split=train...")
    ds = load_dataset("LangAGI-Lab/cactus", split="train")
    logger.info(f"Loaded {len(ds)} rows from the cactus dataset")
    return list(ds)


# Parsing utilities from fine_tuning_cactus.py (kept intact)

def parse_dialogue_field(dialogue_field):
    messages = []

    if isinstance(dialogue_field, (list, tuple)):
        for turn in dialogue_field:
            if isinstance(turn, dict):
                text = None
                role = None
                for k in ("text", "utterance", "utterance_text", "message", "content"):
                    if k in turn:
                        text = turn[k]
                        break
                for k in ("role", "speaker", "sender"):
                    if k in turn:
                        role = str(turn[k])
                        break
                if text is None:
                    vals = [v for v in turn.values() if isinstance(v, str)]
                    text = vals[0] if vals else ""
                role_norm = _normalize_role(role)
                messages.append((role_norm, str(text)))
            else:
                try:
                    speaker, text = turn
                    messages.append((_normalize_role(speaker), str(text)))
                except Exception:
                    messages.append(("user", str(turn)))
        return messages

    if isinstance(dialogue_field, str):
        matches = list(re.finditer(r'([^\n:]{1,50}?):', dialogue_field))
        if matches:
            for i, m in enumerate(matches):
                speaker = m.group(1).strip()
                start = m.end()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(dialogue_field)
                text = dialogue_field[start:end].strip()
                if text:
                    messages.append((_normalize_role(speaker), text))
            return messages

        lines = [ln.strip() for ln in dialogue_field.splitlines() if ln.strip()]
        for ln in lines:
            m = re.match(r"^([^:\-\t]+)[:\-\t]+(.+)$", ln)
            if m:
                speaker = m.group(1).strip()
                text = m.group(2).strip()
                messages.append((_normalize_role(speaker), text))
            else:
                messages.append(("user", ln))
        return messages

    return []


def _normalize_role(role):
    if role is None:
        return "user"
    r = str(role).lower()
    if r in ("assistant", "bot", "agent"):
        return "assistant"
    if r in ("counselor", "counsellor"):
        return "assistant"
    if r in ("therapist", "doctor", "practitioner"):
        return "assistant"
    if r in ("client", "patient", "human", "user"):
        return "user"
    if "user" in r or "human" in r or "client" in r or "patient" in r:
        return "user"
    if "assistant" in r or "bot" in r or "agent" in r or "therapist" in r:
        return "assistant"
    return "user"

############################################
# 3. Unified normalization -> conversations
############################################

# Conversation format used by rolling-window builder:
# {"id": <id>, "turns": [{"role":"user|assistant","text":...}, ...]}

def annomi_df_to_conversations(df: pd.DataFrame) -> List[dict]:
    convs = []
    if "transcript_id" in df.columns:
        groups = df.groupby("transcript_id")
    else:
        groups = [(None, df)]

    for tid, group in groups:
        if "utterance_id" in group.columns:
            group_sorted = group.sort_values(by="utterance_id")
        else:
            group_sorted = group.sort_index()
        group_sorted = group_sorted.reset_index(drop=True)

        turns = []
        for _, row in group_sorted.iterrows():
            # map interlocutor to role
            inter = str(row.get("interlocutor", "")).lower()
            if inter == "client":
                role = "user"
            elif inter == "therapist":
                role = "assistant"
            else:
                role = _normalize_role(inter)

            text = None
            for c in ("utterance_text", "utterance", "text", "message", "content"):
                if c in row and pd.notna(row.get(c)):
                    text = row.get(c)
                    break
            text = "" if text is None else str(text)
            turns.append({"role": role, "text": text})

        convs.append({"id": tid, "turns": turns})
    logger.info(f"Converted AnnoMI into {len(convs)} conversations")
    return convs


def cactus_rows_to_conversations(rows: List[Any]) -> List[dict]:
    convs = []
    for idx, row in enumerate(rows):
        # get dialogue field (handles dict-like objects)
        dialogue = None
        if isinstance(row, dict):
            dialogue = row.get("dialogue")
        else:
            try:
                dialogue = row["dialogue"]
            except Exception:
                dialogue = getattr(row, "dialogue", None)
        if dialogue is None:
            continue
        parsed = parse_dialogue_field(dialogue)
        turns = [{"role": r, "text": t} for (r, t) in parsed]
        convs.append({"id": row.get("id", idx) if isinstance(row, dict) else idx, "turns": turns})
    logger.info(f"Converted cactus into {len(convs)} conversations")
    return convs

############################################
# 4. Rolling-window generator (shared)
############################################

def build_samples_from_conversations(convs: List[dict], window_size: int, include_id: bool = False):
    """Build samples where each sample ends with an assistant reply and the previous
    message is a user message (client->therapist pattern). The window resets per conversation.
    """
    samples = []
    for conv in convs:
        turns = conv.get("turns", [])
        for i in range(len(turns) - 1):
            cur = turns[i]
            nxt = turns[i + 1]
            if cur.get("role") == "user" and nxt.get("role") == "assistant":
                end_idx = i + 1
                start_idx = max(0, end_idx - window_size)
                context = turns[start_idx:end_idx + 1]

                messages = [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    }
                ]

                for r in context[:-1]:
                    role = "user" if r.get("role") == "user" else "assistant"
                    messages.append({"role": role, "content": str(r.get("text", "")).strip()})

                messages.append({"role": "assistant", "content": str(context[-1].get("text", "")).strip()})

                sample = {"messages": messages}
                if include_id:
                    sample["conv_id"] = conv.get("id")
                samples.append(sample)
    return samples


def prepare_dataset_for_window_conversations(convs: List[dict], window_size: int, include_id: bool = True, require_full_window: bool = False):
    logger.info(f" - Building samples with window size {window_size}...")
    samples_w = build_samples_from_conversations(convs, window_size, include_id=include_id)
    logger.info(f"   -> samples before filtering: {len(samples_w)}")

    if require_full_window:
        filtered = [s for s in samples_w if len(s.get("messages", [])) - 2 == window_size]
        logger.info(f"   -> samples after requiring full window ({window_size}): {len(filtered)}")
        return Dataset.from_list(filtered)

    return Dataset.from_list(samples_w)

############################################
# 5. Train/test split utilities
############################################

def train_test_split_convs(convs: List[dict], test_frac: float = 0.1, by_conversation: bool = True, seed: int = 42):
    if not 0.0 < test_frac < 1.0:
        raise ValueError("test_frac must be between 0 and 1")

    if by_conversation:
        idxs = list(range(len(convs)))
        random.Random(seed).shuffle(idxs)
        split_at = int(len(idxs) * (1.0 - test_frac))
        train_idx = set(idxs[:split_at])
        train_convs = [convs[i] for i in range(len(convs)) if i in train_idx]
        test_convs = [convs[i] for i in range(len(convs)) if i not in train_idx]
    else:
        # flatten to turns is complicated; fall back to conv-level split
        return train_test_split_convs(convs, test_frac=test_frac, by_conversation=True, seed=seed)

    logger.info(f"Split dataset: {len(train_convs)} train conversations, {len(test_convs)} test conversations")
    return train_convs, test_convs

############################################
# 6. Model load / eval / train (kept shared)
############################################

def load_model_tokenizer_from_checkpoint(checkpoint_dir: str):
    logger.info(f"Loading model/tokenizer from checkpoint: {checkpoint_dir}")
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint_dir,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=None,
    )
    try:
        tokenizer.model_max_length = MAX_SEQ_LENGTH
    except Exception:
        pass
    return model, tokenizer


def evaluate_model_on_test(checkpoint_dir: str, test_dataset: Dataset, n_examples: int = 5, window_size: int = None):
    model, tokenizer = load_model_tokenizer_from_checkpoint(checkpoint_dir)
    logger.info(f"Evaluating checkpoint {checkpoint_dir} on up to {n_examples} test windows")

    try:
        if window_size is not None:
            samples = [ex for ex in test_dataset if len(ex.get("messages", [])) - 2 == window_size]
        else:
            samples = list(test_dataset)

        count = min(n_examples, len(samples))
        for i in range(count):
            ex = samples[i]
            msgs = ex.get("messages", [])
            if len(msgs) == 0:
                continue

            if msgs[-1].get("role") == "assistant":
                prompt_msgs = msgs[:-1]
                target = msgs[-1].get("content", "")
            else:
                prompt_msgs = msgs
                target = ""

            prompt_text = tokenizer.apply_chat_template(
                prompt_msgs,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
                tools=[],
            )
            device = model.device if hasattr(model, "device") else torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Check full tokenized length (no truncation) and then apply truncation to fit model ctx
            try:
                full_inputs = tokenizer(prompt_text, return_tensors="pt", truncation=False)
                orig_input_tokens = int(full_inputs.input_ids.shape[1])
            except Exception:
                try:
                    orig_input_tokens = len(tokenizer.encode(prompt_text))
                except Exception:
                    orig_input_tokens = None

            inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH)
            input_ids = inputs.input_ids.to(device)

            if orig_input_tokens is not None and orig_input_tokens > MAX_SEQ_LENGTH:
                logger.warning(f"Prompt for eval sample {i} truncated: original_tokens={orig_input_tokens} > MAX_SEQ_LENGTH={MAX_SEQ_LENGTH}")

            # compute input token count and model context early so we can log relation
            try:
                input_tokens = int(input_ids.shape[1])
            except Exception:
                input_tokens = None

            model_ctx = None
            try:
                if hasattr(model, "max_seq_length") and model.max_seq_length is not None:
                    model_ctx = model.max_seq_length
                elif hasattr(model, "config") and hasattr(model.config, "max_position_embeddings"):
                    model_ctx = model.config.max_position_embeddings
            except Exception:
                model_ctx = None
            if model_ctx is None:
                model_ctx = MAX_SEQ_LENGTH

            if input_tokens is not None and model_ctx:
                try:
                    pct = (input_tokens / float(model_ctx)) * 100.0
                except Exception:
                    pct = None
                if pct is not None:
                    logger.info(f"Input tokens: {input_tokens} / {model_ctx} ({pct:.1f}%)")
                else:
                    logger.info(f"Input tokens: {input_tokens} / {model_ctx}")
            else:
                logger.info(f"Input tokens: {input_tokens} (model context: {model_ctx})")

            with torch.no_grad():
                gen_ids = model.generate(
                    input_ids,
                    max_new_tokens=256,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.7,
                )

            try:
                total_tokens_after = int(gen_ids.shape[1])
                generated_tokens = total_tokens_after - (input_tokens or 0)
            except Exception:
                total_tokens_after = None
                generated_tokens = None

            try:
                start_idx = input_tokens
                generated = tokenizer.decode(gen_ids[0][start_idx:], skip_special_tokens=True).strip()
            except Exception:
                generated = tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()

            generated = re.sub(r"<think>.*?</think>\n*", "", generated, flags=re.S)
            generated = re.sub(r"<tool_call>.*?</tool_call>\n*", "", generated, flags=re.S)
            generated = re.sub(r"</?(think|tool_call)[^>]*>\n*", "", generated)
            generated = generated.strip()

            logger.info(f"\n--- Eval sample #{i} prompt ---\n{prompt_text}")
            logger.info(f"\n--- Target (from dataset) ---\n{target}")
            logger.info(f"\n--- Generated response ---\n{generated}")
    except Exception as e:
        logger.exception(f"Evaluation failed: {e}")


class LastTurnOnlyDataCollator(DataCollatorForLanguageModeling):
    """
    Custom DataCollator that finds the LAST occurrence of the assistant's response template
    and masks out (sets to -100) everything before it (the prompt + previous dialogue turns).
    This avoids training on the same past assistant turns multiple times in overlapping windows.
    """
    def __init__(self, tokenizer, *args, **kwargs):
        self.tokenizer = tokenizer
        # Qwen encodes <|im_start|>assistant reliably without the newline issues
        self.response_token_ids = self.tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)
        super().__init__(tokenizer=tokenizer, *args, **kwargs)

    def torch_call(self, examples):
        batch = super().torch_call(examples)
        
        for i in range(len(examples)):
            response_start_idx = None
            labels = batch["labels"][i]
            
            # Find the LAST occurrence of the response template tokens
            for idx in range(len(labels) - len(self.response_token_ids), -1, -1):
                if labels[idx : idx + len(self.response_token_ids)].tolist() == self.response_token_ids:
                    response_start_idx = idx
                    break
            
            if response_start_idx is None:
                # If the template wasn't found, mask the entire sequence
                batch["labels"][i, :] = -100
            else:
                # Mask everything BEFORE the final assistant turn.
                # + length of the template so we don't train on the template tokens themselves.
                # We optionally add +1 to also mask the trailing newline token.
                mask_up_to = response_start_idx + len(self.response_token_ids)
                # Next token is usually \n, mask it too if it's a newline
                if mask_up_to < len(labels) and labels[mask_up_to] == self.tokenizer.encode("\n", add_special_tokens=False)[0]:
                    mask_up_to += 1
                
                batch["labels"][i, : mask_up_to] = -100
                
        return batch


def train_for_window(window_size: int, train_dataset: Dataset, eval_dataset: Dataset = None, output_dir_base: str = OUTPUT_DIR, num_train_epochs: int = 5, eval_steps: int = 100, use_early_stopping: bool = True, early_stopping_patience: int = 3):
    out_dir = f"{output_dir_base}_w{window_size}"
    logger.info(f"\n=== Training run: window_size={window_size} -> output: {out_dir} ===")

    logger.info("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=None,
    )

    logger.info("Applying LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    logger.info("Formatting train dataset with tokenizer...")
    def format_chat_local(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
                tools=[],
            )
        }

    ds_formatted = train_dataset.map(format_chat_local)

    # Clean any thinking/tool tags from the formatted text so they are not used for training
    def _clean_thinking_tags(example):
        txt = example.get("text", "") or ""
        try:
            txt = re.sub(r"<think>.*?</think>\n*", "", txt, flags=re.S)
            txt = re.sub(r"<tool_call>.*?</tool_call>\n*", "", txt, flags=re.S)
            txt = re.sub(r"</?(think|tool_call)[^>]*>\n*", "", txt)
            txt = txt.strip()
        except Exception:
            pass
        return {"text": txt}

    try:
        ds_formatted = ds_formatted.map(_clean_thinking_tags)
    except Exception:
        # If mapping fails, proceed without cleaning (best-effort)
        logger.warning("Failed to clean thinking/tool tags from formatted training dataset")

    # Ensure tokenizer knows the model max length so truncation behaviour is consistent
    try:
        tokenizer.model_max_length = MAX_SEQ_LENGTH
    except Exception:
        pass

    # Compute token-length diagnostics for train dataset (sample-limited)
    def _compute_token_length_stats(dataset, field_name="text", sample_limit=2000):
        lengths = []
        n = 0
        for i, ex in enumerate(dataset):
            if sample_limit and i >= sample_limit:
                break
            txt = ex.get(field_name, "")
            try:
                toks = tokenizer(txt, truncation=False)
                l = int(toks.input_ids.shape[1]) if hasattr(toks, "input_ids") else len(toks["input_ids"])
            except Exception:
                try:
                    l = len(tokenizer.encode(txt))
                except Exception:
                    l = None
            if l is not None:
                lengths.append(l)
            n += 1
        if len(lengths) == 0:
            return {"count": n, "sample_count": 0}
        import statistics

        stats = {
            "sample_count": len(lengths),
            "min": min(lengths),
            "max": max(lengths),
            "mean": statistics.mean(lengths),
            "median": statistics.median(lengths),
            "num_exceeding_max_seq_in_sample": sum(1 for l in lengths if l > MAX_SEQ_LENGTH),
            "sample_lengths": lengths,
        }
        return stats

    # Prepare formatted eval dataset (if provided) so trainer can run evaluation
    ds_eval_formatted = None
    if eval_dataset is not None and len(eval_dataset) > 0:
        try:
            ds_eval_formatted = eval_dataset.map(format_chat_local)
            logger.info(f"Prepared formatted eval dataset (counts: raw={len(eval_dataset)}, formatted={len(ds_eval_formatted)})")
        except Exception as e:
            logger.warning(f"Failed to format eval dataset: {e}")
        try:
            ds_eval_formatted = ds_eval_formatted.map(_clean_thinking_tags)
        except Exception:
            logger.warning("Failed to clean thinking/tool tags from formatted eval dataset")

    # Ensure output directory exists and write initial training metadata
    os.makedirs(out_dir, exist_ok=True)
    try:
        meta_path = os.path.join(out_dir, "training_metadata.json")
        metadata = {
            "window_size": window_size,
            "eval_steps": eval_steps,
            "use_early_stopping": bool(use_early_stopping),
            "early_stopping_patience": early_stopping_patience,
            "num_raw_train_examples": None,
            "num_formatted_train_examples": None,
            "num_raw_eval_examples": None,
            "num_formatted_eval_examples": None,
            "training_args": {
                "per_device_train_batch_size": 2,
                "gradient_accumulation_steps": 4,
                "num_train_epochs": num_train_epochs,
                "learning_rate": 2e-4,
            },
        }
        try:
            metadata["num_raw_train_examples"] = int(len(train_dataset))
        except Exception:
            metadata["num_raw_train_examples"] = None
        try:
            metadata["num_formatted_train_examples"] = int(len(ds_formatted))
        except Exception:
            metadata["num_formatted_train_examples"] = None
        try:
            if eval_dataset is not None:
                metadata["num_raw_eval_examples"] = int(len(eval_dataset))
        except Exception:
            metadata["num_raw_eval_examples"] = None
        try:
            if ds_eval_formatted is not None:
                metadata["num_formatted_eval_examples"] = int(len(ds_eval_formatted))
        except Exception:
            metadata["num_formatted_eval_examples"] = None

        # Add token-length diagnostics (sample-limited) for train and eval formatted texts
        try:
            train_token_stats = _compute_token_length_stats(ds_formatted, field_name="text", sample_limit=2000)
            metadata["train_token_length_stats"] = train_token_stats
        except Exception as e:
            logger.warning(f"Failed to compute train token length stats: {e}")
            metadata["train_token_length_stats"] = {}

        try:
            if ds_eval_formatted is not None:
                eval_token_stats = _compute_token_length_stats(ds_eval_formatted, field_name="text", sample_limit=2000)
            else:
                eval_token_stats = {"sample_count": 0}
            metadata["eval_token_length_stats"] = eval_token_stats
        except Exception as e:
            logger.warning(f"Failed to compute eval token length stats: {e}")
            metadata["eval_token_length_stats"] = {}

        with open(meta_path, "w") as mf:
            json.dump(metadata, mf, indent=2)
        logger.info(f"Wrote initial training metadata to {meta_path}")
    except Exception as e:
        logger.warning(f"Failed to write initial metadata: {e}")

    # Log a few training examples (raw messages + formatted text) for inspection
    try:
        n_log = min(5, len(train_dataset))
    except Exception:
        n_log = 0
    if n_log > 0:
        logger.info(f"Logging {n_log} training examples (raw messages and formatted text):")
        for i in range(n_log):
            try:
                raw = train_dataset[i]
                messages = raw.get("messages", raw.get("text", None))
                conv_id = raw.get("conv_id", raw.get("id", None))
                logger.info(f"--- Train example #{i} conv_id={conv_id} ---")
                if isinstance(messages, list):
                    for m in messages:
                        if isinstance(m, dict):
                            role = m.get("role", "")
                            text = m.get("content", m.get("text", ""))
                        else:
                            role = "user"
                            text = str(m)
                        logger.info(f"  {role}: {text}")
                else:
                    logger.info(f"  messages/raw: {messages}")

                try:
                    fmt = ds_formatted[i].get("text", "")
                    logger.info(f"  formatted text: {fmt}")
                except Exception:
                    fmt = None
            except Exception as e:
                logger.info(f"Failed to log train example #{i}: {e}")

    # Configure evaluation strategy and whether to load best model based on early-stopping flag
    eval_strategy_val = "steps" if use_early_stopping else "no"
    load_best_at_end = bool(use_early_stopping)

    training_args = TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=20,
        num_train_epochs=num_train_epochs,
        learning_rate=2e-4,
        logging_steps=10,
        eval_strategy=eval_strategy_val,
        eval_steps=eval_steps,
        save_steps=eval_steps,
        load_best_model_at_end=load_best_at_end,
        metric_for_best_model="loss",
        greater_is_better=False,
        save_total_limit=3,
        optim="adamw_8bit",
        lr_scheduler_type="linear",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        output_dir=out_dir,
    )

    class EvalLoggingCallback(TrainerCallback):
        """Callback to log evaluation metrics and save evaluation metadata during training."""
        def on_evaluate(self, args, state, control, **kwargs):
            metrics = kwargs.get("metrics", None) or {}
            step = getattr(state, "global_step", None)
            epoch = getattr(state, "epoch", None)
            logger.info(f"Evaluation at step={step}, epoch={epoch}: {metrics}")
            try:
                meta_path = os.path.join(out_dir, "training_metadata.json")
                if os.path.exists(meta_path):
                    with open(meta_path, "r") as mf:
                        metadata = json.load(mf)
                else:
                    metadata = {}

                eval_hist = metadata.get("eval_history", [])
                eval_entry = {"step": step, "epoch": epoch, "metrics": metrics}
                eval_hist.append(eval_entry)
                metadata["eval_history"] = eval_hist
                with open(meta_path, "w") as mf:
                    json.dump(metadata, mf, indent=2)

                step_file = os.path.join(out_dir, f"eval_metadata_step_{step or 'na'}.json")
                with open(step_file, "w") as sf:
                    json.dump(eval_entry, sf, indent=2)
            except Exception as e:
                logger.warning(f"Failed to write evaluation metadata: {e}")

    # Build callbacks depending on whether early stopping is enabled
    callbacks_list = [EvalLoggingCallback()]
    if use_early_stopping:
        callbacks_list.insert(0, EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))

    trainer_kwargs = dict(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds_formatted,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=training_args,
        callbacks=callbacks_list,
        data_collator=LastTurnOnlyDataCollator(
            tokenizer=tokenizer,
            mlm=False
        )
    )
    if ds_eval_formatted is not None:
        trainer_kwargs["eval_dataset"] = ds_eval_formatted

    trainer = SFTTrainer(**trainer_kwargs)

    logger.info("Starting training...")
    trainer.train()

    logger.info(f"Saving LoRA adapter and tokenizer to {out_dir}")
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    # Save trainer log history and update metadata with best checkpoint info
    try:
        meta_path = os.path.join(out_dir, "training_metadata.json")
        # save training log history if available
        log_hist = None
        try:
            log_hist = getattr(trainer.state, "log_history", None)
        except Exception:
            log_hist = None
        if log_hist is not None:
            try:
                log_path = os.path.join(out_dir, "training_log.json")
                with open(log_path, "w") as lf:
                    json.dump(log_hist, lf, indent=2)
                logger.info(f"Saved training log history to {log_path}")
            except Exception as e:
                logger.warning(f"Failed to save training log history: {e}")

        # update metadata with best checkpoint info if available
        try:
            best_ckpt = getattr(trainer.state, "best_model_checkpoint", None)
        except Exception:
            best_ckpt = None
        try:
            if os.path.exists(meta_path):
                with open(meta_path, "r") as mf:
                    metadata = json.load(mf)
            else:
                metadata = {}
            metadata["best_model_checkpoint"] = best_ckpt
            with open(meta_path, "w") as mf:
                json.dump(metadata, mf, indent=2)
            logger.info(f"Updated training metadata with best checkpoint: {best_ckpt}")
        except Exception as e:
            logger.warning(f"Failed to update metadata with best checkpoint: {e}")
    except Exception:
        logger.error("Failed to save training logs or update metadata after training")
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and/or test LoRA models with fixed conversation windows on AnnoMI or Cactus")
    parser.add_argument("--dataset", choices=["annomi", "cactus"], default="annomi", help="Which dataset to use: annomi or cactus")
    parser.add_argument("--annomi-path", type=str, default=DEFAULT_ANNOMI_PATH, help="Path to AnnoMI CSV")
    parser.add_argument("--mode", choices=["train", "test", "all"], default="all", help="Mode: train, test (only), or all")
    parser.add_argument("--windows", nargs="+", type=int, default=[1,2,5], help="Window sizes to run")
    parser.add_argument("--output-base", type=str, default=OUTPUT_DIR, help="Base output dir for checkpoints")
    parser.add_argument("--train-frac", type=float, default=0.90, help="Train fraction for split (e.g., 0.9)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split/shuffle")
    parser.add_argument("--n-test-examples", type=int, default=5, help="Number of test windows to evaluate per model")
    parser.add_argument("--checkpoint-base", type=str, default=OUTPUT_DIR, help="Base path to look for saved checkpoints when testing")
    parser.add_argument("--require-full-window", dest="require_full_window", action="store_true", help="Require full window length when building samples")
    parser.add_argument("--num-train-epochs", type=int, default=5, help="Number of epochs for training")
    parser.add_argument("--eval-steps", type=int, default=100, help="Number of training steps between evaluations when evaluation_strategy='steps'")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience in number of evaluations")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--early-stopping", dest="early_stopping", action="store_true", help="Enable early stopping during training (default)")
    group.add_argument("--no-early-stopping", dest="early_stopping", action="store_false", help="Disable early stopping and run fixed number of epochs")
    parser.set_defaults(early_stopping=True)
    args = parser.parse_args()

    if args.dataset == "annomi":
        df = load_annomi_dataset(args.annomi_path)
        convs = annomi_df_to_conversations(df)
        output_base = args.output_base
    else:
        rows = load_cactus_rows()
        convs = cactus_rows_to_conversations(rows)
        # If user didn't override --output-base, set a sensible default for cactus
        if args.output_base and args.output_base != OUTPUT_DIR:
            output_base = args.output_base
        else:
            output_base = "qwen-cactus-therapist"

    train_convs, eval_convs = train_test_split_convs(convs, test_frac=1.0 - args.train_frac, by_conversation=True, seed=args.seed)

    for w in args.windows:
        logger.info(f"\nPreparing datasets for window size {w}...")
        ds_train = prepare_dataset_for_window_conversations(train_convs, w, include_id=True, require_full_window=args.require_full_window)
        # Use an eval split (default 10%) for in-training evaluation / early stopping
        ds_eval = prepare_dataset_for_window_conversations(eval_convs, w, include_id=True, require_full_window=args.require_full_window)

        if args.mode in ("train", "all"):
            logger.info(f"Starting train for window {w}...")
            notify_start_process("SFT", extra_info={"dataset": args.dataset, "mode": args.mode, "window": w, "num_train_epochs": args.num_train_epochs, "num_train_examples": len(ds_train), "num_eval_examples": len(ds_eval)})
            train_for_window(
                w,
                ds_train,
                ds_eval,
                output_dir_base=output_base,
                num_train_epochs=args.num_train_epochs,
                eval_steps=args.eval_steps,
                use_early_stopping=args.early_stopping,
                early_stopping_patience=args.patience,
            )
            notify_end_process("SFT")

        if args.mode in ("test", "all"):
            # Allow explicit --checkpoint-base, otherwise use the computed output_base
            if args.checkpoint_base and args.checkpoint_base != OUTPUT_DIR:
                checkpoint_base = args.checkpoint_base
            else:
                checkpoint_base = output_base
            checkpoint_dir = f"{checkpoint_base}_w{w}"
            if not os.path.exists(checkpoint_dir):
                logger.info(f"Checkpoint for window {w} not found at {checkpoint_dir}, skipping evaluation.")
            else:
                evaluate_model_on_test(checkpoint_dir, ds_eval, n_examples=args.n_test_examples, window_size=w)
    
