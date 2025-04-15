#!/bin/bash

# Default paths
GRPO_DIR=${GRPO_DIR:-"$HOME/grpo"}
MODELS_DIR=${MODELS_DIR:-"$HOME/pliny_sft/models"}
ENV_FILE=".env"

# Help function
show_help() {
  echo "Usage: ./run.sh [base|lora] [options]"
  echo ""
  echo "Arguments:"
  echo "  base                Run the base training container"
  echo "  lora                Run the LoRA-based training container"
  echo ""
  echo "Options:"
  echo "  --env-file PATH     Path to .env file (default: .env in current directory)"
  echo "  --grpo-dir PATH     Set the path to the GRPO directory (default: ~/grpo)"
  echo "  --models-dir PATH   Set the path to the models directory (default: ~/pliny_sft/models)"
  echo "  --help              Show this help message"
  echo ""
  echo "Environment variables can be set in the .env file with the following format:"
  echo "  WANDB_API_KEY=your_wandb_api_key"
  echo "  OPENAI_API_KEY=your_openai_api_key"
}

# Parse command line arguments
if [ $# -lt 1 ]; then
  show_help
  exit 1
fi

CONTAINER_TYPE=$1
shift

while [ $# -gt 0 ]; do
  case "$1" in
    --env-file)
      ENV_FILE="$2"
      shift 2
      ;;
    --grpo-dir)
      GRPO_DIR="$2"
      shift 2
      ;;
    --models-dir)
      MODELS_DIR="$2"
      shift 2
      ;;
    --help)
      show_help
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      show_help
      exit 1
      ;;
  esac
done

# Check if .env file exists
if [ ! -f "$ENV_FILE" ]; then
  echo "Warning: .env file not found at $ENV_FILE"
  echo "Will use environment variables if set, otherwise container may fail"
else
  # Load environment variables from .env file
  echo "Loading environment variables from $ENV_FILE"
  export $(grep -v '^#' "$ENV_FILE" | xargs)
fi

# Verify required environment variables
if [ -z "$WANDB_API_KEY" ]; then
  echo "Warning: WANDB_API_KEY is not set in .env file or environment"
fi

if [ -z "$OPENAI_API_KEY" ]; then
  echo "Warning: OPENAI_API_KEY is not set in .env file or environment"
fi

# Create directories if they don't exist
mkdir -p "$GRPO_DIR"

# Run the appropriate container
if [ "$CONTAINER_TYPE" == "base" ]; then
  echo "Running base training container..."
  sudo docker run --gpus all \
    -v "$GRPO_DIR":/app/grpo \
    --env-file "$ENV_FILE" \
    training-base
elif [ "$CONTAINER_TYPE" == "lora" ]; then
  echo "Running LoRA training container..."
  mkdir -p "$MODELS_DIR"
  sudo docker run --gpus all \
    -v "$GRPO_DIR":/app/grpo \
    -v "$MODELS_DIR"/qwen-jailbreak-finetuned:/app/pliny_sft/models/qwen-jailbreak-finetuned \
    --env-file "$ENV_FILE" \
    training-lora
else
  echo "Invalid container type: $CONTAINER_TYPE"
  show_help
  exit 1
fi 