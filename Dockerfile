FROM drikster80/vllm-gh200-openai

# Install required packages
RUN pip install unsloth datasets openai wandb

# Set the working directory
WORKDIR /app

# Copy the training script
COPY train_notebook.py .

# Set the entrypoint to run the training script
ENTRYPOINT ["python", "train_notebook.py"]
