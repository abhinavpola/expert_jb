FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Install Python and pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && rm -rf /var/lib/apt/lists/*

# Install required packages
RUN pip install marimo vllm unsloth datasets openai wandb

# Set the working directory
WORKDIR /app

# Copy the training script
COPY train_notebook.py .

# Set the entrypoint to run the training script
ENTRYPOINT ["python", "train_notebook.py"]
