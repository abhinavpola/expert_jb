import os
import torch
import wandb
import random # Added for sampling in callback
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    TrainerCallback,
    logging,
    TrainerState, # Explicitly import TrainerState
    TrainerControl # Explicitly import TrainerControl
)
from peft import LoraConfig, get_peft_model # Removed PeftModel, prepare_model_for_kbit_training import
from trl import SFTTrainer

# --- Configuration ---
model_name = "Qwen/Qwen2.5-7B-Instruct"
# HuggingFace dataset configuration
dataset_name = "jdineen/human-jailbreaks"  # <--- HUGGINGFACE DATASET NAME
dataset_split = "train"                # <--- DATASET SPLIT TO USE
text_column = "text"               # <--- NAME OF THE TEXT COLUMN CONTAINING JAILBREAKS
output_dir = "models/qwen32b-human-jailbreak-finetuned"

os.makedirs(output_dir, exist_ok=True)

wandb_project_name = "qwen-jailbreak-sft"

lora_r = 16
lora_alpha = 32
lora_dropout = 0.05
lora_target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Training config
num_train_epochs = 5
per_device_train_batch_size = 5
gradient_accumulation_steps = 5
gradient_checkpointing = True
max_grad_norm = 0.3
learning_rate = 2e-4
weight_decay = 0.001
optim = "paged_adamw_32bit"
lr_scheduler_type = "cosine"
max_steps = -1
warmup_ratio = 0.03
group_by_length = True
save_steps = 50
logging_steps = 10
max_seq_length = 1024
packing = False

# Generation callback config
callback_logging_steps = 50
callback_num_samples = 2
callback_max_new_tokens = 200
callback_seed = 42

# --- Optional: Weights & Biases Setup ---
use_wandb = True
if use_wandb:
    # Ensure wandb login or WANDB_API_KEY env var is set
    wandb.init(project=wandb_project_name, config={
        "model_name": model_name, "dataset": dataset_name, "lora_r": lora_r,
        "lora_alpha": lora_alpha, "learning_rate": learning_rate, "epochs": num_train_epochs,
        "batch_size": per_device_train_batch_size, "grad_accum": gradient_accumulation_steps,
    })
# -----------------------------------------

# --- Load Dataset ---
print("Loading dataset...")
# Load dataset from Hugging Face
dataset = load_dataset(dataset_name, split=dataset_split)
print(f"Dataset loaded with {len(dataset)} examples.")
train_dataset = dataset

# --- Load Tokenizer ---
print(f"Loading tokenizer for {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("Set pad_token to eos_token")
tokenizer.padding_side = "right"

# --- Formatting Function (using ChatML template and custom prompt) ---
# Formats each data sample using the jailbreak text to create the completion.
def formatting_prompts_func(example):
    output_texts = []
    # Use the configured column name
    texts = example[text_column]

    for i in range(len(texts)):
        text = texts[i] if texts[i] else ""

        if not text:
            print(f"Warning: Skipping row {i} due to missing text.")
            messages = [{"role": "system", "content": "You are a helpful assistant."}]
        else:
            # Construct the prompt
            constructed_prompt = f"I want you to generate a jailbreak prompt"

            # Structure for chat template
            messages = [
                {"role": "user", "content": constructed_prompt},
                {"role": "assistant", "content": text} # The jailbreak text is the desired completion
            ]

        # Apply the chat template
        formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        output_texts.append(formatted_text)
    return output_texts

# --- Load Model ---
print(f"Loading base model: {model_name}...")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True,
)

# --- Prepare Model for PEFT ---
model.config.use_cache = False
model.config.pretraining_tp = 1

# --- LoRA Configuration ---
peft_config = LoraConfig(
    lora_alpha=lora_alpha, lora_dropout=lora_dropout, r=lora_r,
    bias="none", task_type="CAUSAL_LM", target_modules=lora_target_modules
)

# --- Initialize Training Arguments Directly ---
training_arguments = TrainingArguments(
    output_dir=output_dir, num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps, optim=optim,
    logging_steps=logging_steps, learning_rate=learning_rate,
    weight_decay=weight_decay, fp16=False, bf16=True, max_grad_norm=max_grad_norm,
    max_steps=max_steps, warmup_ratio=warmup_ratio, group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type, report_to="wandb" if use_wandb else "none",
    gradient_checkpointing=gradient_checkpointing,
    save_strategy="no",
    push_to_hub=False
)

# --- Custom Callback for Intermediate Generation from Training Set ---
class GenerationCallback(TrainerCallback):
    def __init__(self, tokenizer, dataset, text_column_name, seed=42, num_samples=2, generation_steps=50, max_new_tokens=50, use_wandb=False):
        self.dataset = dataset
        self.text_column = text_column_name # Store the name of the column holding the text
        self.tokenizer = tokenizer
        self.seed = seed
        self.num_samples = num_samples
        self.generation_steps = generation_steps
        self.max_new_tokens = max_new_tokens
        self.use_wandb = use_wandb

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step > 0 and state.global_step % self.generation_steps == 0:
            model = kwargs["model"]
            print(f"\n--- Generating {self.num_samples} sample outputs at step {state.global_step} (using seed {self.seed}) ---")

            model.eval()
            device = next(model.parameters()).device
            generated_texts_log = ""

            if len(self.dataset) > 0 and self.num_samples > 0:
                num_available = len(self.dataset)
                random.seed(self.seed)
                sample_indices = random.sample(range(num_available), min(self.num_samples, num_available))

                for i, index in enumerate(sample_indices):
                    # Get text from the sampled dataset row using the correct column name
                    text = self.dataset[index][self.text_column]
                    if not text:
                        print(f"Skipping sample {i+1} due to empty text at index {index}.")
                        continue

                    # Construct the prompt
                    constructed_prompt = f"I want you to generate a jailbreak prompt"

                    # Simplified print statement
                    print(f"\nSample {i+1}/{self.num_samples} | Prompt (index {index}):\n{constructed_prompt}")

                    # Format the prompt using the chat template for generation
                    messages = [{"role": "user", "content": constructed_prompt}]
                    # Add generation prompt for inference
                    formatted_prompt_for_gen = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs = self.tokenizer(formatted_prompt_for_gen, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_length - self.max_new_tokens).to(device)

                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs, max_new_tokens=self.max_new_tokens,
                            eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.pad_token_id,
                            do_sample=True, temperature=0.7, top_p=0.9
                        )

                    full_decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    # Adjust decoding if needed based on how apply_chat_template formats the input
                    input_decoded = self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
                    # A simple way to find generated text is often to look after the last turn separator
                    generated_text = full_decoded_output[len(input_decoded):].strip()

                    # Simplified print statement
                    print(f"Generated Output:\n{generated_text}")
                    print("-" * 20)

                    # Change: Prepare log entry for wandb as plain text
                    generated_texts_log += f"--- Sample {i+1} (Index: {index}) ---\n"
                    generated_texts_log += f"Prompt:\n{constructed_prompt}\n\n"
                    generated_texts_log += f"Generated Output:\n{generated_text}\n"
                    generated_texts_log += "--------------------------------------------------\n\n"

            else:
                print("Dataset is empty or num_samples is 0, skipping generation.")

            print("--------------------------------------------------")

            # Change: Log plain text to wandb, removing wandb.Html
            if self.use_wandb and wandb.run and generated_texts_log:
                 wandb.log({
                     "step": state.global_step,
                     "generated_samples_text": generated_texts_log # Log as plain text
                 })

            model.train()


# --- Initialize Trainer ---
print("Initializing SFTTrainer...")
generation_callback = GenerationCallback(
    dataset=train_dataset,
    text_column_name=text_column, # Pass the name of the text column
    tokenizer=tokenizer,
    seed=callback_seed,
    num_samples=callback_num_samples,
    generation_steps=callback_logging_steps,
    max_new_tokens=callback_max_new_tokens,
    use_wandb=use_wandb
)

trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=train_dataset,
    peft_config=peft_config,
    formatting_func=formatting_prompts_func, # Use the updated formatting function
    callbacks=[generation_callback]
)

# --- Start Training ---
print("Starting training...")
trainer.train()
print("Training finished.")

# --- Save Final Model ---
print(f"Saving final LoRA adapter to {output_dir}...")
trainer.save_model(output_dir)
print("Adapter saved.")

if use_wandb:
    wandb.finish()
