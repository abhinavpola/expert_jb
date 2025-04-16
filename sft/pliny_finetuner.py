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
model_name = "Qwen/Qwen2.5-32B-Instruct"
# Assumes CSV has columns: filename, header, content
dataset_path = "jailbreaks.csv" # <--- PATH TO YOUR CSV FILE (filename,header,content)
filename_column = "filename"             # <--- NAME OF THE FILENAME COLUMN (e.g., name.mkd)
content_column = "content"               # <--- NAME OF THE CONTENT COLUMN (the desired jailbreak)
# Note: 'header' column from CSV is not used in this script based on the request.
output_dir = "models/qwen32b-jailbreak-finetuned"

os.makedirs(output_dir, exist_ok=True)

wandb_project_name = "qwen-jailbreak-sft"

lora_r = 16
lora_alpha = 32
lora_dropout = 0.05
lora_target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# # BitsAndBytes config (REMOVED - Was commented out)
# use_4bit = True
# bnb_4bit_quant_type = "nf4"
# bnb_4bit_compute_dtype = torch.bfloat16
# use_nested_quant = False

# Training config
num_train_epochs = 5
per_device_train_batch_size = 2
gradient_accumulation_steps = 4
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
callback_max_new_tokens = 100
callback_seed = 42

# --- Optional: Weights & Biases Setup ---
use_wandb = True
if use_wandb:
    # Ensure wandb login or WANDB_API_KEY env var is set
    wandb.init(project=wandb_project_name, config={
        "model_name": model_name, "dataset": dataset_path, "lora_r": lora_r,
        "lora_alpha": lora_alpha, "learning_rate": learning_rate, "epochs": num_train_epochs,
        "batch_size": per_device_train_batch_size, "grad_accum": gradient_accumulation_steps,
    })
# -----------------------------------------

# --- Load Dataset ---
print("Loading dataset...")
# Load dataset assuming columns 'filename', 'header', 'content'
dataset = load_dataset("csv", data_files=dataset_path, split="train")
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
# Formats each data sample using the filename to create the prompt.
def formatting_prompts_func(example):
    output_texts = []
    # Use the configured column names
    filenames = example[filename_column]
    contents = example[content_column]

    for i in range(len(filenames)):
        filename = filenames[i] if filenames[i] else ""
        content = contents[i] if contents[i] else ""

        if not filename or not content:
             print(f"Warning: Skipping row {i} due to missing filename or content.")
             messages = [{"role": "system", "content": "You are a helpful assistant."}]
        else:
            # Extract name from filename (remove .mkd extension)
            name = os.path.splitext(filename)[0] # More robust way to remove extension
            # Construct the prompt based on the template
            constructed_prompt = f"I want you to generate a jailbreak prompt for {name}"

            # Structure for chat template
            messages = [
                {"role": "user", "content": constructed_prompt},
                {"role": "assistant", "content": content} # The content is the desired completion
            ]

        # Apply the chat template
        formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        output_texts.append(formatted_text)
    return output_texts

# --- Load Model ---
print(f"Loading base model: {model_name}...")
# compute_dtype = getattr(torch, bnb_4bit_compute_dtype) # REMOVED - Was commented out
# bnb_config = BitsAndBytesConfig(...) # REMOVED - Was commented out
# Check GPU compatibility with bfloat16 - REMOVED - Was commented out

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # quantization_config=bnb_config, # REMOVED - Was commented out
    device_map="auto",
    trust_remote_code=True,
    # torch_dtype=torch.bfloat16 # Optionally specify dtype if not using quantization
)

# --- Prepare Model for PEFT ---
model.config.use_cache = False
model.config.pretraining_tp = 1
# model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing) # REMOVED - Was commented out

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
    # Takes filename_column_name now, which corresponds to the original prompt source
    def __init__(self, tokenizer, dataset, filename_column_name, seed=42, num_samples=2, generation_steps=50, max_new_tokens=50, use_wandb=False):
        self.dataset = dataset
        self.filename_column = filename_column_name # Store the name of the column holding the filename
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
                    # Get filename from the sampled dataset row using the correct column name
                    filename = self.dataset[index][self.filename_column]
                    if not filename:
                        print(f"Skipping sample {i+1} due to empty filename at index {index}.")
                        continue

                    # Construct the prompt using the same logic as formatting_prompts_func
                    name = os.path.splitext(filename)[0]
                    constructed_prompt = f"I want you to generate a jailbreak prompt for {name}"

                    # Simplified print statement
                    print(f"\nSample {i+1}/{self.num_samples} | Prompt (from {filename}, index {index}):\n{constructed_prompt}")

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
                    # For Qwen, this might be '<|im_end|>\n<|im_start|>assistant\n' or similar
                    # Let's try removing the input prompt string again, might need adjustment
                    generated_text = full_decoded_output[len(input_decoded):].strip()
                    # Fallback/alternative: find the last known part of the prompt template
                    # assistant_marker = "assistant\n" # Adjust based on actual template output
                    # try:
                    #     gen_start_index = full_decoded_output.rindex(assistant_marker) + len(assistant_marker)
                    #     generated_text = full_decoded_output[gen_start_index:].strip()
                    # except ValueError:
                    #      generated_text = "Error decoding - check template markers" # Handle case where marker isn't found

                    # Simplified print statement
                    print(f"Generated Output:\n{generated_text}")
                    print("-" * 20)

                    # Change: Prepare log entry for wandb as plain text
                    generated_texts_log += f"--- Sample {i+1} (Filename: {filename}, Index: {index}) ---\n"
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
    filename_column_name=filename_column, # Pass the name of the filename column
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
