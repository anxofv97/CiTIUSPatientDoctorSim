FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install minimal build tools for wheels (optional but helpful for some deps)
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies if requirements.txt exists
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install -r /app/requirements.txt

# Copy project
COPY . /app

EXPOSE 6416

# Run deploy_eeyore.py directly with Python (no shell script)
CMD ["python", "./eeyore_code/deploy_eeyore.py", "--host", "0.0.0.0", "--port", "6416", "--temperature", "1.0", "--top-p", "0.8", "--max-new-tokens", "512", "--sequence-bias", "[[[128009], -4.0]]", "--exponential-decay-length-penalty", "0", "1.01", "--load-in-8bit"]
