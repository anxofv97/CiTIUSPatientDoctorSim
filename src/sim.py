import logging
import logging.config
import json
import asyncio
import re
import os
import unicodedata
from datetime import datetime

from .doctor import Doctor
from .patient import Patient
from .llm import LLM
from .prompts import DOCTOR_INSTRUCTIONS
from .eeyore_prompt import prepare_prompt_from_profile, create_cognitive_system_prompt

logger = logging.getLogger(__name__)

def generate_patient_prompt(profile_id: int, profile_path: str = "data/test_profile_cognitive_model.json", expand_history: bool = False) -> str:
    """Generate a patient system prompt for a given profile ID."""
    # Load test profile and prepare patient system prompt
    with open(profile_path, "r") as f:
        test_set = json.load(f)

    data = test_set[profile_id]
    data.pop("conversation", None)

    # Build patient system prompt
    _, profile, profile_dict = asyncio.run(prepare_prompt_from_profile(data))
    patient_sys_prompt, profile, profile_dict = asyncio.run(create_cognitive_system_prompt(data, profile, profile_dict))

    if expand_history:
        # Get cognitive models section from the patient system prompt
        pattern = r"Patient History:(.*?)In the upcoming conversation"
        match = re.search(pattern, patient_sys_prompt, re.DOTALL)
        if match:
            patient_history = match.group(1)
        else:
            logger.warning("Patient History section not found in patient system prompt.")
            patient_history = ""
        # Run prompt to expand patient history
        expand_instructions = "Elaborate a backstory (4-5 paragraphs) for the patient that is compatible with both the patient's history and the cognitive models. You can add more details about the patient's life, personality, relationships, and experiences, as long as they don't contradict the information already available."
        model = LLM()
        response = model.run_prompt(patient_history, instructions=expand_instructions, max_new_tokens=4096)
        logger.info(f"Expanded patient history: {response[1].strip()}")
        # Replace patient history section in the patient system prompt with the expanded version
        new_section = response[1].strip()
        updated_text = re.sub(
            pattern,
            f"Patient History: {new_section}\n\nIn the upcoming conversation",
            patient_sys_prompt,
            flags=re.DOTALL
        )
        patient_sys_prompt = updated_text
    return patient_sys_prompt

def clean_text(text):
    """
    Clean and normalize text. Ensure that no Unicode characters or special characters that could cause issues in downstream processing remain. Also normalize whitespace.
    """
    # Explicit replacements
    replacements = {
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "—": "-",
        "–": "-",
    }

    for k, v in replacements.items():
        text = text.replace(k, v)

    # Normalize unicode
    text = unicodedata.normalize("NFKD", text)

    # Remove non-ASCII characters
    text = text.encode("ascii", "ignore").decode("ascii")

    # Clean extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

def save_simulation(output_path, profile_id, max_turns, patient, doctor, turns_completed, init_timestamp, end_timestamp, doctor_model: str = None, expand_history: bool = False, stopping_detection: bool = True, doctor_instructions: str = None, patient_sys_prompt: str = None):
    # If output_path doesn't exist, create it
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # Collect histories and build dump
    doctor_history = doctor.get_conversation_history()

    # Calculate elapsed time
    time_elapsed = (end_timestamp - init_timestamp).total_seconds()

    dump = {
        "profile_id": profile_id,
        "timestamp_init": init_timestamp.isoformat() + "Z",
        "timestamp_end": end_timestamp.isoformat() + "Z",
        "time_elapsed_seconds": time_elapsed,
        "turns_max": max_turns,
        "turns_completed": turns_completed,
        "doctor_model": doctor_model or getattr(doctor.model, "model_name", None),
        "expand_history": expand_history,
        "stopping_detection": stopping_detection,
        "instructions_doctor": doctor_instructions,
        "instructions_patient": patient_sys_prompt,
        "history_patient": patient.get_conversation_history(),
        "history_doctor": doctor_history,
    }

    fname = f"{output_path}/conversation_profile{profile_id}_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}"

    # Write JSON dump
    try:
        with open(fname+".json", "w", encoding="utf-8") as jf:
            json.dump(dump, jf, ensure_ascii=False, indent=2, sort_keys=True)
        logger.info(f"Simulation dumped to {fname}")
    except Exception:
        logger.exception("Failed to write simulation dump")

    # Save original conversation to TXT (and keep text in memory)
    try:
        original_txt = fname + ".txt"
        original_lines = []
        with open(original_txt, "w", encoding="ascii") as f:
            for msg in dump.get("doctor_history", []):
                if msg.get("role") == "system":
                    continue
                if msg.get("role") == "assistant":
                    speaker = "Clinician"
                elif msg.get("role") == "user":
                    speaker = "Client"
                else:
                    continue
                content = clean_text(msg.get("content", ""))
                line = f"{speaker}: {content}\n\n"
                original_lines.append(line)
                f.write(line)
        original_text = "".join(original_lines)
        logger.info(f"Original conversation dumped to {original_txt}")
    except Exception:
        logger.exception("Failed to write conversation dump (original)")

def run_doctor_patient_conversation(doctor_instructions, patient_url: str = "http://127.0.0.1:6416/v1/chat/completions", profile_id: int = 0, doctor_model: str = "qwen3", expand_history: bool = False, max_turns: int = 10, output_path: str = "data_out", stopping_detection: bool = True):
    """Run an automated conversation between the doctor LLM and the patient endpoint.

    - The doctor uses the local `LLM` (`doctor.get_response`) with `DOCTOR_INSTRUCTIONS_TEST` as system prompt.
    - The patient is driven by the HTTP chat completions endpoint and receives a system prompt generated
      from the profile in `data/test_profile_cognitive_model.json`.

    Args:
        patient_url: patient endpoint URL.
        profile_id: index into the test profile JSON to use for the patient system prompt.
        turns: number of doctor->patient->doctor exchanges to perform.
    """
    logger.info(f"Starting doctor-patient conversation simulation. Profile_ID: {profile_id}")
    patient_sys_prompt = generate_patient_prompt(profile_id, expand_history=expand_history)

    logger.info(f"Patient instructions: {patient_sys_prompt}")
    logger.info(f"Doctor instructions: {doctor_instructions}")

    # Allow providing a local or HF model identifier for the doctor LLM
    if doctor_model:
        doctor = Doctor(model_name=doctor_model, max_new_tokens=256, stopping_detection=stopping_detection)
    else:
        doctor = Doctor(max_new_tokens=256, stopping_detection=stopping_detection)
    patient = Patient(patient_url=patient_url)

    # Configure doctor LLM with its system instructions
    doctor_greeting = "Hello, how are you feeling today? Any concerns or feelings you'd like to share? I'm here to listen and help with whatever you're going through."
    doctor.reset_conversation(instructions=doctor_instructions, doctor_greeting=doctor_greeting)
    patient.reset_conversation(instructions=patient_sys_prompt)

    # Capture init timestamp
    init_timestamp = datetime.utcnow()

    # Initial doctor greeting
    doc_msg = doctor_greeting
    logger.info(f"Doctor greeting: {doc_msg.strip()}")
    # Alternate messages between patient (HTTP) and doctor (local LLM)
    for i in range(max_turns):
        # Send doctor's message to patient endpoint
        try:
            pat_msg = patient.get_response(doc_msg)
            logger.info(f"Patient: {pat_msg.strip()}")
        except Exception:
            logger.exception("Error getting response from patient endpoint")
            break
        # Send patient reply to doctor LLM
        try:
            _, doc_reply = doctor.get_response(pat_msg)
        except Exception:
            logger.exception("Error getting response from doctor LLM")
            break
        logger.info(f"Doctor: {doc_reply.strip()}")
        doc_msg = doc_reply
        if doctor.is_conversation_ended():
            break

    # Capture end timestamp
    end_timestamp = datetime.utcnow()

    logger.info("Conversation ended.")

    save_simulation(output_path, profile_id, max_turns, patient, doctor, turns_completed=i+1, init_timestamp=init_timestamp, end_timestamp=end_timestamp, doctor_model=doctor_model, expand_history=expand_history, stopping_detection=stopping_detection, doctor_instructions=doctor_instructions, patient_sys_prompt=patient_sys_prompt)

    