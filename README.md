# ROGERS

A simulation and evaluation framework for automated doctor-patient conversations.

The repository runs therapy-style dialogues between a local doctor LLM and a patient endpoint, supports fine-tuning data preparation, and includes evaluation tooling for generated conversation outputs.

This is the implementation for the paper "ROGERS: Generating Synthetic Therapy Sessions using LLM Therapists Aligned with CBT and MI principles" accepted at Information Processing & Management Conference 2026, Wuhan, China. 

## Key components

- `main.py`
  - Primary experiment driver for running doctor-patient conversation simulations.
  - Currently configured to execute a performance test using a selected local model.

- `fine_tuning.py`
  - Data loading and fine-tuning dataset preparation for AnnoMI and Cactus-style conversational data.
  - Uses `unsloth`, `trl`, and `transformers` to prepare datasets and train models.

- `evaluate_folder.py`
  - Evaluates generated conversations in a folder using prompt-based scoring templates from `prompts_evaluation/`.
  - Outputs per-conversation JSONL results and a summary JSON.

- `src/`
  - `src/sim.py` — conversation orchestration, patient prompt generation, and dump persistence.
  - `src/llm.py` — local Qwen3 model wrapper and prompt generation utilities.
  - `src/patient.py` — patient endpoint client and optional local Eeyore service launcher.
  - `src/doctor.py` — doctor agent wrapper with stopping detection and response generation.
  - `src/prompts.py` — doctor instruction prompts used by the simulation and fine-tuning workflows.
  - `src/notify.py` — optional Telegram notification helper.

## Prerequisites

- Linux environment
- Python 3.x (the repository includes `eeyore_env/` which may already contain a working Python environment)
- GPU with CUDA support if running local transformer inference and fine-tuning
- Slurm Workload Manager installed and configured (although it can be run independently)
- Required Python packages in `requirements.txt`

## Installation

1. Activate the existing environment if available:

```bash
source eeyore_env/bin/activate
```

2. Otherwise create and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Verify the environment is active and all dependencies are installed.

## Data and outputs

- `data/`
  - `AnnoMI-full.csv`, `AnnoMI-simple.csv` — psychotherapy and conversational training data.
  - `test_profile_cognitive_model.json` — patient profile definitions used for generating patient system prompts.

- `prompts_evaluation/`
  - Prompt templates used by `evaluate_folder.py` to score conversations.

- `data_out/`
  - Generated conversation outputs and experiment folders.

- `data_eval/`
  - Evaluation results produced by `evaluate_folder.py`.

- `models_1e/`, `models_3e/`
  - Local fine-tuned model directories referenced by simulation workflows.

## Usage

### Run simulations

The recommended entrypoint is the SLURM driver script:

```bash
sbatch slurm_main.sh
```

`slurm_main.sh` activates `eeyore_env`, sets locale variables, and runs `python main.py` on the configured GPU partition.

`main.py` contains several experiment blocks. To run a different experiment, edit `main.py` and uncomment the desired section before submitting the SLURM job.

Example experiment types in `main.py`:

- Zero-shot doctor instruction variants
- One-shot prompts
- SFT (supervised fine-tuning) model inference with local directories like `models_1e/...` and `models_3e/...`
- Performance test output to `data_out/performance_test`

### Run evaluation

The recommended evaluation driver is:

```bash
sbatch slurm_run_evals.sh
```

`slurm_run_evals.sh` activates the environment and runs one or more `evaluate_folder.py` commands. Uncomment the desired evaluation line in the script to evaluate a specific experiment folder.

### Run fine-tuning / dataset prep

Use the fine-tuning SLURM script:

```bash
sbatch slurm_fine_tuning.sh
```

`slurm_fine_tuning.sh` activates `eeyore_env` and runs `python fine_tuning.py` with the configured training parameters.

### Direct invocation (for debugging)

If you need to run an individual script outside SLURM for debugging, the commands are:

```bash
python main.py
python evaluate_folder.py --folder data_out/<experiment_folder> --out data_eval/<result_file>.jsonl --model qwen3
python fine_tuning.py
```

This repository is designed to use the SLURM scripts as the normal workflow.

## Local patient endpoint

The patient simulation uses a local Eeyore service endpoint by default at `http://127.0.0.1:6416/v1/chat/completions`.

- The `Patient` class in `src/patient.py` will auto-start a local Eeyore server if the URL points to localhost.
- The helper script `deploy_eeyore.sh` can also start the Eeyore service explicitly.

## Configuration

- `logging_config.json` — logging configuration for the Python scripts.
- `src/prompts.py` — contains the active doctor instruction templates used in experiments.
- `data/test_profile_cognitive_model.json` — patient profile prompt content used to generate the patient system prompt.
- `src/notify.py` uses `TG_TOKEN` and `TG_CHAT_ID` environment variables to send Telegram notifications if configured.

## Notes

- The doctor model can be either a local model directory (e.g. `models_3e/qwen-cactus-therapist_w5`) or a Hugging Face model identifier.
- Generated conversation outputs are stored in JSON and TXT formats under `data_out/`.
- Fine-tuned models can be found in our HF organization, both with [AnnoMI](https://huggingface.co/citiusLTL/qwen-annomi-therapist_w5) and [Cactus](https://huggingface.co/citiusLTL/qwen-cactus-therapist_w5) datasets.

## Reference

If you use this repository, please cite our paper:

```
@inproceedings{villoch2026rogers,
  title = {ROGERS: Generating Synthetic Therapy Sessions using LLM Therapists Aligned with CBT and MI principles},
  author = {Fern{\'a}ndez-Villoch, Anxo and Fern{\'a}ndez-Pichel, Marcos and Piette, John D. and Losada, David E},
  booktitle = {Information Processing \& Management Conference 2026},
  year = {2026},
}
```


## License

This project is released under the MIT License.
