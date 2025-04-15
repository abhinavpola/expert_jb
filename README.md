

# Expert Jailbreaker

This is an attempt to train a model (Llama 4 Scout) using reinforcement learning (GRPO) to jailbreak other models. The project includes tools for fine-tuning models with LoRA adapters and evaluating their jailbreaking capabilities.

## Project Structure

- `pliny_sft/`: Contains scripts for supervised fine-tuning and evaluation
  - `finetuner.py`: Fine-tunes a model with LoRA on jailbreak examples
  - `eval.py`: Evaluates fine-tuned models against target models
- `train_notebook.py`: Base GRPO training script
- `train_sft_notebook.py`: GRPO training script that uses a pre-fine-tuned model
- Docker configuration files for different training scenarios
- Shell scripts for building and running Docker containers

## Usage

### 1. Fine-tuning with LoRA (Optional)

To fine-tune a model with LoRA on jailbreak examples:

```bash
cd pliny_sft
python finetuner.py
```

This will:
- Load the base model (Qwen2.5-7B-Instruct by default)
- Apply LoRA fine-tuning using examples from `jailbreaks.csv`
- Save the adapter to `pliny_sft/models/qwen-jailbreak-finetuned/`

You can modify parameters in `finetuner.py` to adjust the training process.

### 2. Evaluating Fine-tuned Models

To evaluate a fine-tuned model's jailbreaking capabilities:

```bash
cd pliny_sft
python eval.py
```

This will:
- Load the base model with the LoRA adapter
- Generate jailbreak prompts for harmful requests in `eval.csv`
- Test these jailbreaks against target models via OpenRouter
- Log results to `evaluation_log.txt`

### 3. GRPO Training

You can run GRPO training either with a base model or with a fine-tuned model using Docker.

#### Setup

First, create a `.env` file with your API keys:

```
WANDB_API_KEY=your_wandb_api_key
OPENAI_API_KEY=your_openai_api_key
```

#### Building Docker Images

Build the Docker images using the provided script:

```bash
# Build the base training image (without LoRA)
./build.sh base

# Build the LoRA training image (with pre-fine-tuned model)
./build.sh lora
```

#### Running Training

Run training using the provided script:

```bash
# Run base training (without LoRA)
./run.sh base

# Run training with LoRA adapter
./run.sh lora --models-dir /path/to/pliny_sft/models
```

Additional options:
```
--env-file PATH     Path to .env file (default: .env in current directory)
--grpo-dir PATH     Set the path to the GRPO directory (default: ~/grpo)
--models-dir PATH   Set the path to the models directory (default: ~/pliny_sft/models)
```

## Differences Between Base and LoRA Training

### Base Training (`train_notebook.py`)
- Starts with a vanilla model
- Applies GRPO directly to the base model
- Suitable for initial exploration

### LoRA Training (`train_sft_notebook.py`)
- Starts with a model that has already been fine-tuned on jailbreak examples
- Loads the LoRA adapter before applying GRPO
- Potentially more effective as the model has prior exposure to jailbreak patterns

## Memory Optimization

If you encounter OOM errors on your GPU:
- Reduce batch size
- Enable gradient accumulation
- Use a smaller model or reduce LoRA rank
- Enable gradient checkpointing

## Notes

- The current implementation may OOM on a single H100. Optimizations are in progress.
- For best results, fine-tune on high-quality jailbreak examples before GRPO training.
- The evaluation script can help identify which jailbreak patterns are most effective.
