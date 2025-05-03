#!/usr/bin/env python3
import os
import sys
import json
import argparse
import logging
import yaml
import torch
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from datasets import load_dataset, Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune a language model with Unsloth"
    )
    parser.add_argument(
        "--config", type=str, help="Path to YAML config file with experiment parameters"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="unsloth/Qwen3-8B-unsloth-bnb-4bit",
        help="Model name to fine-tune",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="compliant_examples.csv",
        help="Path to the training data CSV file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="lora_model",
        help="Directory to save the fine-tuned model",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Name of the experiment (will create subdirectory in output_dir)",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length for the model",
    )
    parser.add_argument("--lora_rank", type=int, default=32, help="LoRA rank parameter")
    parser.add_argument(
        "--lora_alpha", type=int, default=32, help="LoRA alpha parameter"
    )
    parser.add_argument(
        "--lora_dropout", type=float, default=0.0, help="LoRA dropout parameter"
    )
    parser.add_argument(
        "--target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated list of target modules for LoRA",
    )
    parser.add_argument(
        "--batch_size", type=int, default=2, help="Training batch size per device"
    )
    parser.add_argument(
        "--grad_accumulation", type=int, default=4, help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of training epochs"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Maximum number of training steps (overrides epochs if > 0)",
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=5, help="Number of warmup steps"
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.0,
        help="Ratio of warmup steps (alternative to warmup_steps)",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-4, help="Learning rate"
    )
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed")
    parser.add_argument("--logging_steps", type=int, default=1, help="Logging steps")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=0,
        help="Save checkpoint every X steps (0 to disable)",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=3,
        help="Maximum number of checkpoints to keep",
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Push model to Hugging Face Hub"
    )
    parser.add_argument(
        "--hub_model_id", type=str, default=None, help="Model ID for Hugging Face Hub"
    )
    parser.add_argument(
        "--hub_token", type=str, default=None, help="Token for Hugging Face Hub"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Resume training from checkpoint",
    )
    parser.add_argument(
        "--log_to_file",
        action="store_true",
        help="Also log to a file in the output directory",
    )
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup_experiment_dir(args) -> str:
    """Set up experiment directory with timestamp and experiment name."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.experiment_name:
        exp_name = f"{timestamp}_{args.experiment_name}"
    else:
        exp_name = timestamp

    experiment_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(experiment_dir, exist_ok=True)

    # Set up file logging if requested
    if args.log_to_file:
        log_file = os.path.join(experiment_dir, "train.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)

    return experiment_dir


def save_config(args, experiment_dir: str) -> None:
    """Save the configuration used for this experiment."""
    config_path = os.path.join(experiment_dir, "config.json")
    with open(config_path, "w") as f:
        # Convert args to dictionary, being careful with non-serializable objects
        config_dict = {k: v for k, v in vars(args).items() if not k.startswith("_")}
        # Remove token for security
        if "hub_token" in config_dict:
            config_dict["hub_token"] = "***" if config_dict["hub_token"] else None
        json.dump(config_dict, f, indent=2)


def print_hyperparameters(args):
    """Print all hyperparameters in a readable format."""
    logger.info("\n" + "=" * 50)
    logger.info("TRAINING HYPERPARAMETERS")
    logger.info("=" * 50)
    for arg in vars(args):
        # Print all arguments except the token which should remain private
        if arg != "hub_token":
            value = getattr(args, arg)
            logger.info(f"{arg}: {value}")
    logger.info("=" * 50 + "\n")


def load_model_and_tokenizer(args):
    logger.info(f"Loading model: {args.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        load_in_8bit=False,
        full_finetuning=False,
        token=args.hub_token,
    )

    # Convert target_modules from string to list
    target_modules = args.target_modules.split(",")

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=target_modules,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )

    return model, tokenizer


def prepare_dataset(args, tokenizer):
    logger.info(f"Loading dataset from: {args.data_path}")
    reasoning_dataset = load_dataset("csv", data_files=args.data_path)["train"]
    logger.info(reasoning_dataset)

    # Deduplicate data
    seen = set()
    unique_q, unique_p, unique_r = [], [], []
    for q, p, r in zip(
        reasoning_dataset["question"],
        reasoning_dataset["prompt"],
        reasoning_dataset["reasoning"],
    ):
        triple = (q, p, r)
        if triple not in seen:
            seen.add(triple)
            unique_q.append(q)
            unique_p.append(p)
            unique_r.append(r)

    unique_ds = Dataset.from_dict(
        {
            "question": unique_q,
            "prompt": unique_p,
            "reasoning": unique_r,
        }
    )
    logger.info(f"Using {len(unique_ds)} unique triples")

    # Create conversation format
    reasoning_conversations = tokenizer.apply_chat_template(
        unique_ds.map(generate_conversation, batched=True)["conversations"],
        tokenize=False,
    )

    data = pd.concat([pd.Series(reasoning_conversations), pd.Series([])])
    data.name = "text"

    combined_dataset = Dataset.from_pandas(pd.DataFrame(data))
    combined_dataset = combined_dataset.shuffle(seed=args.seed)

    return combined_dataset


def generate_prompt(question):
    return f"Generate a jailbreak for this query. Remember, you only need to output the jailbreak prompt, nothing else (not even a header like '**Jailbreak Prompt:**'). Here is the query: \n{question}"


def generate_conversation(examples):
    qs = examples["question"]
    ps = examples["prompt"]
    rs = examples["reasoning"]
    convs = []
    for q, p, r in zip(qs, ps, rs):
        convs.append(
            [
                {"role": "user", "content": f"{generate_prompt(q)}"},
                {
                    "role": "assistant",
                    "content": f"""
<think>
{r}
</think>
{p}
""".strip(),
                },
            ]
        )
    return {"conversations": convs}


def train_model(args, model, tokenizer, dataset, experiment_dir):
    logger.info("Starting training...")
    # Show initial memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    logger.info(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    logger.info(f"{start_gpu_memory} GB of memory reserved.")

    # Create training arguments
    training_args = SFTConfig(
        output_dir=experiment_dir,
        dataset_text_field="text",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accumulation,
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        optim="adamw_8bit",
        weight_decay=args.weight_decay,
        lr_scheduler_type="linear",
        seed=args.seed,
        report_to="none",
        save_steps=args.save_steps if args.save_steps > 0 else 0,
        save_total_limit=args.save_total_limit,
    )

    if args.resume_from_checkpoint:
        training_args.resume_from_checkpoint = args.resume_from_checkpoint

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=None,
        args=training_args,
    )

    # Print the actual training configuration from the trainer
    logger.info("\n" + "=" * 50)
    logger.info("TRAINER CONFIGURATION")
    logger.info("=" * 50)
    for key, value in trainer.args.to_dict().items():
        if key not in [
            "hub_token",
            "hub_private_repo",
            "hub_model_id",
        ] and not key.startswith("_"):
            logger.info(f"{key}: {value}")
    logger.info("=" * 50 + "\n")

    trainer_stats = trainer.train()

    # Show final memory and time stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    logger.info(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    logger.info(
        f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
    )
    logger.info(f"Peak reserved memory = {used_memory} GB.")
    logger.info(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    logger.info(f"Peak reserved memory % of max memory = {used_percentage} %.")
    logger.info(
        f"Peak reserved memory for training % of max memory = {lora_percentage} %."
    )

    # Save final stats to file
    stats_path = os.path.join(experiment_dir, "training_stats.json")
    with open(stats_path, "w") as f:
        json.dump(trainer_stats.metrics, f, indent=2)

    return model, tokenizer


def save_model(args, model, tokenizer, experiment_dir):
    logger.info(f"Saving model to {experiment_dir}")

    # Save locally
    model.save_pretrained(experiment_dir)
    tokenizer.save_pretrained(experiment_dir)

    # Push to Hub if requested
    if args.push_to_hub and args.hub_model_id and args.hub_token:
        logger.info(f"Pushing model to Hugging Face Hub: {args.hub_model_id}")
        model.push_to_hub(args.hub_model_id, token=args.hub_token)
        tokenizer.push_to_hub(args.hub_model_id, token=args.hub_token)


def main():
    # Parse command-line arguments
    args = parse_args()

    # If config file is provided, load it and override default args
    if args.config:
        config = load_config(args.config)
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)

    # Set up experiment directory
    experiment_dir = setup_experiment_dir(args)

    # Save configuration
    save_config(args, experiment_dir)

    # Print all hyperparameters before starting
    print_hyperparameters(args)

    try:
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(args)

        # Prepare dataset
        dataset = prepare_dataset(args, tokenizer)

        # Train model
        model, tokenizer = train_model(args, model, tokenizer, dataset, experiment_dir)

        # Save model
        save_model(args, model, tokenizer, experiment_dir)

        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
