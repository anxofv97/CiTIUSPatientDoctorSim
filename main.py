import json
import logging
import logging.config

from src.sim import run_doctor_patient_conversation
from src.prompts import DOCTOR_INSTRUCTIONS, DOCTOR_INSTRUCTIONS_1, DOCTOR_INSTRUCTIONS_2, DOCTOR_INSTRUCTIONS_ONE_SHOT, DOCTOR_SFT
from src.notify import notify_start_process, notify_end_process

# Config logging from file
def setup_logging(config_file="./logging_config.json"):
    with open(config_file, "r") as f:
        config = json.load(f)
    logging.config.dictConfig(config)

setup_logging()
logger = logging.getLogger(__name__)

EEYORE_NUM_PROFILES = 12
MAX_TURNS = 100
NUM_PROFILE_REPEATS = 9

if __name__=="__main__":
    # ZERO SHOT
    notify_start_process("ZERO SHOT - doctor_instructions")
    for _ in range(NUM_PROFILE_REPEATS):
        for i in range(EEYORE_NUM_PROFILES):
            run_doctor_patient_conversation(max_turns=MAX_TURNS, profile_id=i, output_path=f"data_out/zero_shot_doctor_instructions", 
                                        doctor_model="qwen3",
                                        doctor_instructions=DOCTOR_INSTRUCTIONS, expand_history=False, stopping_detection=True)
    notify_end_process("ZERO SHOT - doctor_instructions")
    notify_start_process("ZERO SHOT - doctor_instructions_1")
    for _ in range(NUM_PROFILE_REPEATS):
        for i in range(EEYORE_NUM_PROFILES):
            run_doctor_patient_conversation(max_turns=MAX_TURNS, profile_id=i, output_path=f"data_out/zero_shot_doctor_instructions_1", 
                                        doctor_model="qwen3",
                                        doctor_instructions=DOCTOR_INSTRUCTIONS_1, expand_history=False, stopping_detection=True)
    notify_end_process("ZERO SHOT - doctor_instructions_1")
    notify_start_process("ZERO SHOT - doctor_instructions_2")
    for _ in range(NUM_PROFILE_REPEATS):
        for i in range(EEYORE_NUM_PROFILES):
            run_doctor_patient_conversation(max_turns=MAX_TURNS, profile_id=i, output_path=f"data_out/zero_shot_doctor_instructions_2", 
                                        doctor_model="qwen3",
                                        doctor_instructions=DOCTOR_INSTRUCTIONS_2, expand_history=False, stopping_detection=True)
    notify_end_process("ZERO SHOT - doctor_instructions_2")
    notify_start_process("ZERO SHOT - expand_2")
    for _ in range(NUM_PROFILE_REPEATS):
        for i in range(EEYORE_NUM_PROFILES):
            run_doctor_patient_conversation(max_turns=MAX_TURNS, profile_id=i, output_path=f"data_out/zero_shot_expand_2", 
                                        doctor_model="qwen3",
                                        doctor_instructions=DOCTOR_INSTRUCTIONS_2, expand_history=True, stopping_detection=True)
    notify_end_process("ZERO SHOT - expand_2")
    notify_start_process("ONE SHOT")
    for _ in range(NUM_PROFILE_REPEATS):
        for i in range(EEYORE_NUM_PROFILES):
            run_doctor_patient_conversation(max_turns=MAX_TURNS, profile_id=i, output_path=f"data_out/one_shot", 
                                        doctor_model="qwen3",
                                        doctor_instructions=DOCTOR_INSTRUCTIONS_ONE_SHOT, expand_history=False, stopping_detection=True)
    notify_end_process("ONE SHOT")
    notify_start_process("ONE SHOT - expand")
    for _ in range(NUM_PROFILE_REPEATS):
        for i in range(EEYORE_NUM_PROFILES):
            run_doctor_patient_conversation(max_turns=MAX_TURNS, profile_id=i, output_path=f"data_out/one_shot_expand", 
                                        doctor_model="qwen3",
                                        doctor_instructions=DOCTOR_INSTRUCTIONS_ONE_SHOT, expand_history=True, stopping_detection=True)
    notify_end_process("ONE SHOT - expand")
    # AnnoMI SFT - 1 epoch models
    for w in [1, 5]:
        notify_start_process(f"SFT AnnoMI-1e-{w}")
        for _ in range(NUM_PROFILE_REPEATS):
            for i in range(EEYORE_NUM_PROFILES):
                run_doctor_patient_conversation(max_turns=MAX_TURNS, profile_id=i, output_path=f"data_out/models_1e_annomi_sft_{w}", 
                                            doctor_instructions=DOCTOR_SFT, expand_history=True, doctor_model=f"models/models_1e/qwen-annomi-therapist_w{w}",
                                            stopping_detection=True)
        notify_end_process(f"SFT AnnoMI-1e-{w}")
    # AnnoMI SFT - 3 epoch models
    for w in [1, 5]:
        notify_start_process(f"SFT AnnoMI-3e-{w}")
        for _ in range(NUM_PROFILE_REPEATS):
            for i in range(EEYORE_NUM_PROFILES):
                run_doctor_patient_conversation(max_turns=MAX_TURNS, profile_id=i, output_path=f"data_out/models_3e_annomi_sft_{w}", 
                                            doctor_instructions=DOCTOR_SFT, expand_history=True, doctor_model=f"models/models_3e/qwen-annomi-therapist_w{w}",
                                            stopping_detection=True)
        notify_end_process(f"SFT AnnoMI-3e-{w}")
    # Cactus SFT - 1 epoch models
    for w in [1, 5, 50]:
        notify_start_process(f"SFT Cactus-1e-{w}")
        for _ in range(NUM_PROFILE_REPEATS):
            for i in range(EEYORE_NUM_PROFILES):
                run_doctor_patient_conversation(max_turns=MAX_TURNS, profile_id=i, output_path=f"data_out/models_1e_cactus_sft_{w}", 
                                            doctor_instructions=DOCTOR_SFT, expand_history=True, doctor_model=f"models/models_1e/qwen-cactus-therapist_w{w}",
                                            stopping_detection=True)
        notify_end_process(f"SFT Cactus-1e-{w}")
    # Cactus SFT - 3 epoch model
    for w in [5]:
        notify_start_process(f"SFT Cactus-3e-{w}")
        for _ in range(NUM_PROFILE_REPEATS):
            for i in range(EEYORE_NUM_PROFILES):
                run_doctor_patient_conversation(max_turns=MAX_TURNS, profile_id=i, output_path=f"data_out/models_3e_cactus_sft_{w}", 
                                            doctor_instructions=DOCTOR_SFT, expand_history=True, doctor_model=f"models/models_3e/qwen-cactus-therapist_w{w}",
                                            stopping_detection=True)
        notify_end_process(f"SFT Cactus-3e-{w}")

