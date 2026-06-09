from typing import Tuple, List, Optional
import atexit
import logging
import requests
import socket
import threading
import time
from argparse import Namespace

from starlette.applications import Starlette
from starlette.routing import Route
import uvicorn

from eeyore_code import deploy_eeyore

logger = logging.getLogger(__name__)

DEFAULT_EEYORE_HOST = "127.0.0.1"
DEFAULT_EEYORE_PORT = 6416
DEFAULT_EEYORE_PATH = "/v1/chat/completions"
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_EEYORE_MODEL = "liusiyang/eeyore_sft_epoch2_dpo_round2_epoch1_llama3.1_8B"
DEFAULT_LOAD_IN_8BIT = True
DEFAULT_EEYORE_LOG_LEVEL = "INFO"
DEFAULT_EEYORE_UVICORN_LOG_LEVEL = "debug"
DEFAULT_EEYORE_NO_FILE_LOGGING = True

_local_eeyore_server: Optional["EeyoreServer"] = None
_local_eeyore_lock = threading.Lock()


def _is_local_eeyore_url(patient_url: str) -> bool:
    return patient_url.startswith("http://127.0.0.1") or patient_url.startswith("http://localhost")


class EeyoreServer:
    def __init__(
        self,
        host: str = DEFAULT_EEYORE_HOST,
        port: int = DEFAULT_EEYORE_PORT,
        model: str = DEFAULT_EEYORE_MODEL,
        device_map: str = "auto",
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.8,
        sequence_bias: str = "[[[128009], -4.0]]",
        exponential_decay_length_penalty: tuple[float, float] = (0, 1.01),
        load_in_8bit: bool = DEFAULT_LOAD_IN_8BIT,
        offload_folder: str = "offload",
        uvicorn_log_level: str = DEFAULT_EEYORE_UVICORN_LOG_LEVEL,
        log_level: str = DEFAULT_EEYORE_LOG_LEVEL,
        log_file: str = "eeyore_debug.log",
        no_file_logging: bool = DEFAULT_EEYORE_NO_FILE_LOGGING,
    ):
        self.host = host
        self.port = port
        self.model = model
        self.device_map = device_map
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.sequence_bias = sequence_bias
        self.exponential_decay_length_penalty = exponential_decay_length_penalty
        self.load_in_8bit = load_in_8bit
        self.offload_folder = offload_folder
        self.uvicorn_log_level = uvicorn_log_level
        self.log_level = log_level
        self.log_file = log_file
        self.no_file_logging = no_file_logging

        self.app: Optional[Starlette] = None
        self.server: Optional[uvicorn.Server] = None
        self.thread: Optional[threading.Thread] = None
        self.started = False

    def _create_args_namespace(self) -> Namespace:
        return Namespace(
            host=self.host,
            port=self.port,
            model=self.model,
            device_map=self.device_map,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            sequence_bias=self.sequence_bias,
            exponential_decay_length_penalty=list(self.exponential_decay_length_penalty),
            load_in_8bit=self.load_in_8bit,
            offload_folder=self.offload_folder,
            uvicorn_log_level=self.uvicorn_log_level,
            log_level=self.log_level,
            log_file=self.log_file,
            no_file_logging=self.no_file_logging,
        )

    def _wait_service_ready(self, timeout: int = 120) -> None:
        deadline = time.time() + timeout
        logger.info(f"Waiting for Eeyore service on {self.host}:{self.port}...")
        while time.time() < deadline:
            try:
                with socket.create_connection((self.host, self.port), timeout=3):
                    logger.info("Eeyore service is ready.")
                    return
            except OSError:
                time.sleep(1)
        raise RuntimeError(f"Eeyore service did not become available within {timeout} seconds.")

    def start(self, timeout: int = 120) -> None:
        if self.started:
            return

        self.args = self._create_args_namespace()
        deploy_eeyore.args = self.args

        logger.info("Initializing local Eeyore model...")
        if not deploy_eeyore.initialize_model(
            self.args.model,
            self.args.device_map,
            load_in_8bit=self.args.load_in_8bit,
            offload_folder=self.args.offload_folder,
        ):
            raise RuntimeError("Failed to initialize Eeyore model")

        self.app = Starlette(
            routes=[
                Route("/v1/chat/completions", deploy_eeyore.homepage, methods=["POST"]),
                Route("/health", deploy_eeyore.health_check, methods=["GET"]),
            ],
            on_startup=[deploy_eeyore.startup_event],
        )
        deploy_eeyore.app = self.app

        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level=self.args.uvicorn_log_level,
            access_log=False,
            loop="asyncio",
            lifespan="on",
        )
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()

        self._wait_service_ready(timeout)
        self.started = True
        logger.info("Local Eeyore server started successfully.")

        if not self.no_file_logging:
            atexit.register(self.stop)

    def stop(self) -> None:
        if not self.started or self.server is None:
            return

        logger.info("Stopping local Eeyore server...")
        self.server.should_exit = True
        self.server.force_exit = True
        if self.thread is not None:
            self.thread.join(timeout=30)
        self.started = False
        logger.info("Local Eeyore server stopped.")


def get_local_eeyore_server(
    host: str = DEFAULT_EEYORE_HOST,
    port: int = DEFAULT_EEYORE_PORT,
    model: str = DEFAULT_EEYORE_MODEL,
    load_in_8bit: bool = DEFAULT_LOAD_IN_8BIT,
    no_file_logging: bool = DEFAULT_EEYORE_NO_FILE_LOGGING,
) -> EeyoreServer:
    global _local_eeyore_server
    with _local_eeyore_lock:
        if _local_eeyore_server is None:
            _local_eeyore_server = EeyoreServer(
                host=host,
                port=port,
                model=model,
                load_in_8bit=load_in_8bit,
                no_file_logging=no_file_logging,
            )
            _local_eeyore_server.start()
        return _local_eeyore_server


class Patient:
    """
    Class to simulate a Patient in a therapy session.

    The patient sends conversation history to a local Eeyore server and returns
    the assistant response.
    """

    def __init__(
        self,
        patient_url: Optional[str] = None,
        host: str = DEFAULT_EEYORE_HOST,
        port: int = DEFAULT_EEYORE_PORT,
        api_key: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        start_local: bool = True,
        eeyore_model: str = DEFAULT_EEYORE_MODEL,
        load_in_8bit: bool = DEFAULT_LOAD_IN_8BIT,
        no_file_logging: bool = DEFAULT_EEYORE_NO_FILE_LOGGING,
    ):
        if patient_url is not None:
            self.patient_url = patient_url
        else:
            self.patient_url = f"http://{host}:{port}{DEFAULT_EEYORE_PATH}"

        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()
        self.patient_messages: Optional[List[dict]] = None

        if start_local and _is_local_eeyore_url(self.patient_url):
            get_local_eeyore_server(
                host=host,
                port=port,
                model=eeyore_model,
                load_in_8bit=load_in_8bit,
                no_file_logging=no_file_logging,
            )

    def get_response(self, doctor_message: str) -> str:
        """
        Generate a patient response to the doctor's message.

        Args:
            doctor_message: the latest message from the doctor to respond to.

        Returns:
            The patient's response message content.
        """
        if self.patient_messages is None:
            raise ValueError("Conversation not initialized. Please call reset_conversation() first.")

        self.patient_messages.append({"role": "user", "content": doctor_message})
        payload = {"messages": self.patient_messages}

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            response = self.session.post(
                self.patient_url,
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.error("Failed to call local Eeyore server at %s", self.patient_url)
            logger.exception(exc)
            raise

        try:
            response_data = response.json()
        except ValueError as exc:
            logger.error("Invalid JSON response from Eeyore server: %s", response.text)
            raise RuntimeError("Invalid JSON returned by Eeyore server") from exc

        if not isinstance(response_data, dict) or "choices" not in response_data:
            logger.error("Unexpected response shape from Eeyore server: %s", response_data)
            raise RuntimeError("Unexpected response from Eeyore server")

        choices = response_data.get("choices")
        if not choices or not isinstance(choices, list):
            logger.error("No choices returned by Eeyore server: %s", response_data)
            raise RuntimeError("No completion choices returned by Eeyore server")

        choice = choices[0]
        if "message" not in choice or "content" not in choice["message"]:
            logger.error("Invalid choice format from Eeyore server: %s", choice)
            raise RuntimeError("Invalid completion format from Eeyore server")

        pat_msg = str(choice["message"]["content"]).strip()
        self.patient_messages.append({"role": "assistant", "content": pat_msg})

        return pat_msg

    def reset_conversation(self, instructions: Optional[str] = None):
        """
        Reset/set the conversation state.

        Args:
            instructions: system instructions text (optional)
        """
        self.patient_messages = []
        if instructions:
            self.patient_messages.append({"role": "system", "content": instructions})

    def get_conversation_history(self) -> List[dict]:
        """
        Get the current conversation history as a list of messages.

        Returns:
            A list of message dicts with "role" and "content" keys.
        """
        if self.patient_messages is None:
            raise ValueError("Conversation not initialized. Please call reset_conversation() first.")
        return self.patient_messages
