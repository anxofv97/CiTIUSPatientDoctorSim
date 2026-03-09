import logging
import logging.config
import json
import sys

from sim import interactive_chat

# Config logging from file
def setup_logging(config_file="./logging_config.json"):
    with open(config_file, "r") as f:
        config = json.load(f)
    logging.config.dictConfig(config)
setup_logging()

if __name__=="__main__":
    id_arg = 0
    url_arg = "http://127.0.0.1:6416/v1/chat/completions"
    if len(sys.argv) > 1:
        try:
            id_arg = int(sys.argv[1])
        except Exception:
            pass
    if len(sys.argv) > 2:
        try:
            url_arg = sys.argv[2]
        except Exception:
            pass
    interactive_chat(id=id_arg, url=url_arg)