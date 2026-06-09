from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import asyncio
import json
import uvicorn
import logging
import traceback
from datetime import datetime
import argparse

# Configure initial logging (will be reconfigured in main based on args)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for configuration
pipe = None
args = None

def initialize_model(model_name, device_map="auto", load_in_8bit=False, offload_folder=None):
    """Initialize the model pipeline with optional 8-bit loading."""
    global pipe
    try:
        logger.info(f"Loading model pipeline: {model_name} load_in_8bit={load_in_8bit}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)

        # Determine torch dtype: prefer fp16 on GPU
        torch_dtype = torch.float16 if torch.cuda.is_available() else None

        # Prepare quantization config for bitsandbytes (preferred API)
        quant_config = None
        if load_in_8bit:
            try:
                quant_config = BitsAndBytesConfig(load_in_8bit=True)
            except Exception:
                logger.warning("BitsAndBytesConfig unavailable or bitsandbytes not installed; falling back to no quantization")
                quant_config = None

        # Load model (use quantization_config when provided)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            quantization_config=quant_config,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            offload_folder=offload_folder,
            trust_remote_code=True
        )

        pipe = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map=device_map,
            trust_remote_code=True
        )

        logger.info("Model pipeline loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.error(traceback.format_exc())
        pipe = None
        return False

async def homepage(request):
    try:
        logger.debug(f"Received request: {request.method} {request.url}")
        logger.debug(f"Request headers: {dict(request.headers)}")
        
        # Parse JSON from request body
        try:
            payload = await request.json()
            logger.debug(f"Request payload: {payload}")
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)
        except Exception as e:
            logger.error(f"Error reading request body: {e}")
            return JSONResponse({"error": "Error reading request"}, status_code=400)
        
        # Extract messages from OpenAI format
        messages = payload.get('messages', [])
        if not messages:
            logger.warning("No messages found in request")
            return JSONResponse({"error": "No messages provided"}, status_code=400)
        
        # Validate message format
        if not isinstance(messages, list):
            logger.warning("Messages must be a list")
            return JSONResponse({"error": "Messages must be a list"}, status_code=400)
        
        # Validate each message has required fields
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                logger.warning(f"Invalid message format at index {i}: {msg}")
                return JSONResponse({"error": f"Invalid message format at index {i}"}, status_code=400)
        
        logger.debug(f"Processing request with {len(messages)} messages")
        logger.debug(f"Messages: {messages}")
        
        response_q = asyncio.Queue()
        await request.app.model_queue.put((messages, response_q))
        logger.debug("Request queued for processing")
        
        output = await response_q.get()
        logger.debug(f"Generated response length: {len(str(output))}")
        logger.debug(f"Response: {output}")
        
        # Format response as OpenAI chat completion
        import time
        import uuid
        
        # Handle error responses
        if isinstance(output, dict) and 'error' in output:
            return JSONResponse({
                "error": {
                    "message": output['error'],
                    "type": "server_error",
                    "code": 500
                }
            }, status_code=500)
        
        # Create OpenAI-compatible response
        openai_response = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "eeyore_sft_epoch2_dpo_round1_epoch1_llama3.1_8B",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": str(output)
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": sum(len(str(msg.get('content', '')).split()) for msg in messages),  # Approximate
                "completion_tokens": len(str(output).split()),  # Approximate
                "total_tokens": sum(len(str(msg.get('content', '')).split()) for msg in messages) + len(str(output).split())
            }
        }
        
        return JSONResponse(openai_response)
        
    except Exception as e:
        logger.error(f"Unexpected error in homepage: {e}")
        logger.error(traceback.format_exc())
        return JSONResponse({
            "error": {
                "message": "Internal server error",
                "type": "server_error",
                "code": 500
            }
        }, status_code=500)

async def server_loop(q):
    global pipe
    logger.info("Starting server loop")
    
    while True:
        try:
            (messages, response_q) = await q.get()
            logger.debug(f"Processing {len(messages)} messages")
            
            if pipe is None:
                logger.error("Model pipeline not available")
                await response_q.put({"error": "Model not loaded"})
                continue
            
            start_time = datetime.now()
            logger.debug("Starting model inference...")
            
            # Pass messages directly to the pipeline
            logger.debug(f"Messages: {messages}")
            
            # Parse sequence_bias from JSON string if available
            sequence_bias = None
            if args and args.sequence_bias:
                try:
                    sequence_bias = json.loads(args.sequence_bias)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid sequence_bias JSON: {args.sequence_bias}")
                    sequence_bias = None # Default fallback
            else:
                sequence_bias = None  # Default fallback
            
            # Get generation parameters from args or defaults
            max_tokens = args.max_new_tokens if args else 4096
            exp_decay = tuple(args.exponential_decay_length_penalty) if args else None
            temperature = args.temperature if args else 1
            top_p = args.top_p if args else 0.8
            
            response = pipe(
                messages, 
                max_new_tokens=max_tokens, 
                exponential_decay_length_penalty=exp_decay, 
                sequence_bias=sequence_bias,
                temperature=temperature,
                top_p=top_p
            )
            logger.debug(f"Model response: {response}")
            
            # Extract the generated content from the response
            generated_content = response[0]["generated_text"][-1]["content"]
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            logger.info(f"[eeyore] inference completed in {processing_time:.2f} s. Generated tokens: {len(generated_content.split())}")
            
            
            await response_q.put(generated_content)
            logger.debug("Response sent to queue")
            
        except Exception as e:
            logger.error(f"Error in server loop: {e}")
            logger.error(traceback.format_exc())
            try:
                await response_q.put({"error": f"Processing error: {str(e)}"})
            except:
                pass

async def startup_event():
    logger.info("Starting up application...")
    q = asyncio.Queue()
    app.model_queue = q
    asyncio.create_task(server_loop(q))
    logger.info("Application startup complete")

# Add a health check endpoint for debugging
async def health_check(request):
    logger.debug("Health check requested")
    return JSONResponse({
        "status": "healthy",
        "model_loaded": pipe is not None,
        "timestamp": datetime.now().isoformat()
    })

def parse_arguments():
    """Parse command line arguments for the Eeyore API server"""
    parser = argparse.ArgumentParser(
        description="Eeyore API Server - OpenAI-compatible chat completion API",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Server configuration
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind the server to"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=6416,
        help="Port to bind the server to"
    )
    
    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="liusiyang/eeyore_sft_epoch2_dpo_round2_epoch1_llama3.1_8B",
        help="Model name or path to load"
    )
    
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="Device mapping for model loading (auto, cpu, cuda, etc.)"
    )
    
    # Generation parameters
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,  # Reduced from 4096 for faster inference
        help="Maximum number of tokens to generate"
    )
    
    parser.add_argument(
        "--exponential-decay-length-penalty",
        type=float,
        nargs=2,
        default=[0, 1.01],
        metavar=("START", "DECAY"),
        help="Exponential decay length penalty parameters (start, decay)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=1,
        help="Temperature for model generation"
    )
    
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.8,
        help="Top-p for model generation"
    )
    
    parser.add_argument(
        "--sequence-bias",
        type=str,
        default="[[[128009], -4.0]]",
        help="Sequence bias as JSON string (e.g., '[[[128009], -4.0]]')"
    )
    
    # Logging configuration
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="DEBUG",
        help="Logging level"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        default="eeyore_debug.log",
        help="Log file path"
    )
    
    parser.add_argument(
        "--no-file-logging",
        action="store_true",
        help="Disable file logging (console only)"
    )
    
    # Server options
    parser.add_argument(
        "--uvicorn-log-level",
        type=str,
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        default="debug",
        help="Uvicorn log level"
    )
    
    
    
    # 8-bit and offload options
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load model in 8-bit using bitsandbytes (requires bitsandbytes installed)"
    )

    parser.add_argument(
        "--offload-folder",
        type=str,
        default="offload",
        help="Folder path to use for CPU/NVMe offloading"
    )

    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments and set global variable
    args = parse_arguments()
    globals()['args'] = args  # Make args available globally
    
    # Configure logging based on arguments
    log_handlers = []
    if not args.no_file_logging:
        log_handlers.append(logging.FileHandler(args.log_file))
    log_handlers.append(logging.StreamHandler())
    
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=log_handlers,
        force=True  # Reconfigure logging
    )
    
    logger.info("Initializing Eeyore API Server...")
    logger.info(f"Server configuration: host={args.host}, port={args.port}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Device map: {args.device_map}")
    logger.info(f"Max tokens: {args.max_new_tokens}")
    logger.info(f"Exponential decay: {args.exponential_decay_length_penalty}")
    logger.info(f"Sequence bias: {args.sequence_bias}")
    
    # Initialize the model (support optional 8-bit loading and offload)
    if not initialize_model(args.model, args.device_map, load_in_8bit=args.load_in_8bit, offload_folder=args.offload_folder):
        logger.error("Failed to initialize model. Exiting.")
        exit(1)
    
    app = Starlette(
        routes=[
            Route("/v1/chat/completions", homepage, methods=["POST"]),
            Route("/health", health_check, methods=["GET"]),  # Debug endpoint
        ],
        on_startup=[startup_event],  # Register the startup event
    )
    
    logger.info("Starting uvicorn server...")
    uvicorn.run(
        app, 
        host=args.host, 
        port=args.port, 
        log_level=args.uvicorn_log_level
    )