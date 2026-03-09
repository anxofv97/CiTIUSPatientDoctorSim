import logging
import logging.config
import json
import sys
from datetime import datetime

from src.sim import run_doctor_patient_conversation, generate_patient_prompt
from src.prompts import DOCTOR_INSTRUCTIONS, DOCTOR_INSTRUCTIONS_1, DOCTOR_INSTRUCTIONS_2
# Config logging from file
def setup_logging(config_file="./logging_config.json"):
    with open(config_file, "r") as f:
        config = json.load(f)
    logging.config.dictConfig(config)
setup_logging()
logger = logging.getLogger(__name__)

if __name__=="__main__":
    """for _ in range(1): # 3 runs per profile
        for i in range(12): # 12 different profiles
            run_doctor_patient_conversation(max_turns=100, profile_id=i, output_path="data_out/expand_1", doctor_instructions=DOCTOR_INSTRUCTIONS, expand_history=True)
        for i in range(12): # 12 different profiles
            run_doctor_patient_conversation(max_turns=100, profile_id=i, output_path="data_out/expand_2", doctor_instructions=DOCTOR_INSTRUCTIONS_2, expand_history=True)"""

    for _ in range(10):
        run_doctor_patient_conversation(max_turns=250, profile_id=2, output_path="data_out/expand_2", doctor_instructions=DOCTOR_INSTRUCTIONS_2, expand_history=True)
        run_doctor_patient_conversation(max_turns=250, profile_id=3, output_path="data_out/expand_2", doctor_instructions=DOCTOR_INSTRUCTIONS_2, expand_history=True)