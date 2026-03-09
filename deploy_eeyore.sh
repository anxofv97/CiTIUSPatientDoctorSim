python ./eeyore_code/deploy_eeyore.py \
  --host 127.0.0.1 \
  --port 6416 \
  --temperature 1.0 \
  --top-p 0.8 \
  --max-new-tokens 4096 \
  --sequence-bias "[[[128009], -4.0]]" \
  --exponential-decay-length-penalty 0 1.01