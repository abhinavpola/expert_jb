import os
import torch
import random
import pandas as pd
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from openai import OpenAI # Use OpenAI client for OpenRouter
from dotenv import load_dotenv
import logging

# --- Configuration ---
BASE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_PATH = "models/qwen-jailbreak-finetuned" # Path to your LoRA adapter
DATASET_PATH = "eval.csv" # <--- Use the new CSV
REQUEST_COLUMN = "request_description" # <--- Column with the harmful request text
FILENAME_COLUMN = "filename" # Column with the original filename (used for context/logging)

# Evaluation Config
NUM_SAMPLES = 5 # How many harmful requests/filenames to test
MAX_NEW_TOKENS_LOCAL = 250 # Max tokens for local model jailbreak generation
MAX_NEW_TOKENS_API = 250 # Max tokens for API model response to the jailbreak
SEED = 42 # For reproducibility of sample selection
TEMPERATURE = 0.7 # Generation temperature
TOP_P = 0.9 # Generation top_p

# OpenRouter Models to Test Against the Generated Jailbreak
# Find model IDs at https://openrouter.ai/models
OPENROUTER_MODELS = [
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
    "anthropic/claude-3.5-sonnet",
    # Add more models as needed
]
# No LOCAL_MODEL_ID needed in the list of models to test *against*

# --- Setup ---
load_dotenv() # Load environment variables from .env file
random.seed(SEED)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Load Fine-tuned Model and Tokenizer ---
def load_finetuned_model(base_model_name, adapter_path):
    """Loads the base model and applies the PEFT adapter."""
    logging.info(f"Loading base model: {base_model_name}")
    # Optional: Configure quantization if needed
    # bnb_config = BitsAndBytesConfig(...)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        # quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logging.info("Set pad_token to eos_token")
    tokenizer.padding_side = "right"

    logging.info(f"Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    logging.info("Fine-tuned model and tokenizer loaded.")
    return model, tokenizer

# --- Load Dataset and Prepare Initial Prompts & Context ---
def load_initial_prompts_and_context(dataset_path, filename_col, request_col, num_samples):
    """Loads dataset, samples filenames to create initial prompts, and keeps harmful requests for context."""
    logging.info(f"Loading dataset from: {dataset_path}")
    try:
        dataset = load_dataset("csv", data_files=dataset_path, split="train")
        logging.info(f"Dataset loaded with {len(dataset)} examples.")
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        return [], [], []

    if num_samples > len(dataset):
        logging.warning(f"Requested {num_samples} samples, but dataset only has {len(dataset)}. Using all.")
        num_samples = len(dataset)

    sample_indices = random.sample(range(len(dataset)), num_samples)
    initial_prompts = []
    harmful_requests_context = []
    source_filenames = []

    for index in sample_indices:
        filename = dataset[index][filename_col]
        harmful_request = dataset[index][request_col]

        if not filename:
            logging.warning(f"Skipping sample at index {index} due to empty filename.")
            continue
        if not harmful_request:
             logging.warning(f"Skipping sample at index {index} (Filename: {filename}) due to empty request description.")
             continue

        try:
            # Create the initial prompt for the local model based on filename
            name = os.path.splitext(filename)[0]
            constructed_initial_prompt = f"I want you to generate a jailbreak prompt for {name}"
            initial_prompts.append(constructed_initial_prompt)
            harmful_requests_context.append(harmful_request) # Keep the original request for logging/results
            source_filenames.append(filename)
        except Exception as e:
             logging.warning(f"Error processing filename '{filename}' at index {index}: {e}")

    logging.info(f"Prepared {len(initial_prompts)} initial prompts for local generation.")
    return initial_prompts, harmful_requests_context, source_filenames

# --- Generation Functions (Unchanged) ---
def generate_local(model, tokenizer, prompt, max_new_tokens, temperature, top_p):
    """Generates text using the local fine-tuned model."""
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                do_sample=True, temperature=temperature, top_p=top_p
            )
        input_length = inputs['input_ids'].shape[1]
        generated_ids = outputs[0, input_length:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return generated_text
    except Exception as e:
        logging.error(f"Error during local generation for prompt '{prompt[:50]}...': {e}")
        return f"[ERROR: {e}]"

def generate_openrouter(client, model_id, prompt, max_new_tokens, temperature, top_p):
    """Generates text using an OpenRouter model via API."""
    messages = [{"role": "user", "content": prompt}] # The prompt here will be the generated jailbreak
    try:
        response = client.chat.completions.create(
            model=model_id, messages=messages, max_tokens=max_new_tokens,
            temperature=temperature, top_p=top_p,
            extra_headers={
                "HTTP-Referer": "http://localhost/jailbreak-eval",
                "X-Title": "Jailbreak Eval Script V3 (Pipe)",
            }
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error calling OpenRouter API for model {model_id} and prompt '{prompt[:50]}...': {e}")
        return f"[API ERROR: {e}]"

# --- Main Evaluation ---
if __name__ == "__main__":
    # Load API Key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logging.error("OPENROUTER_API_KEY not found in environment variables or .env file.")
        exit(1)

    # Initialize OpenRouter Client
    openrouter_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    logging.info("OpenRouter client initialized.")

    # Load Local Model (only once)
    local_model, tokenizer = None, None
    try:
        local_model, tokenizer = load_finetuned_model(BASE_MODEL_NAME, ADAPTER_PATH)
    except Exception as e:
        logging.error(f"Failed to load local model: {e}. Cannot proceed.")
        exit(1) # Exit if local model fails, as it's needed to generate jailbreaks

    # Load Initial Prompts and Harmful Request Context
    initial_prompts, harmful_requests_context, source_filenames = load_initial_prompts_and_context(
        DATASET_PATH, FILENAME_COLUMN, REQUEST_COLUMN, NUM_SAMPLES
    )
    if not initial_prompts:
        logging.error("No initial prompts were generated from the dataset. Exiting.")
        exit(1)

    # Run Evaluations
    results = []
    # Iterate through the initial prompts and context
    for i, (initial_prompt, harmful_request, filename) in enumerate(zip(initial_prompts, harmful_requests_context, source_filenames)):
        logging.info(f"\n--- Evaluating Sample {i+1}/{len(initial_prompts)} (Source File: {filename}) ---")
        logging.info(f"Harmful Request Context: {harmful_request}")
        logging.info(f"Initial Prompt for Local Model: {initial_prompt}")

        # --- Step 1: Generate the jailbreak prompt using the local model ---
        logging.info(f"Generating jailbreak with local model ({ADAPTER_PATH})...")
        generated_jailbreak = generate_local(
            local_model, tokenizer, initial_prompt, MAX_NEW_TOKENS_LOCAL, TEMPERATURE, TOP_P
        )
        logging.info(f"Locally Generated Jailbreak:\n{generated_jailbreak}\n" + "-"*20)

        # Store results, including the initial prompt, context, and the generated jailbreak
        result_entry = {
            "source_filename": filename,
            "harmful_request_context": harmful_request,
            "initial_prompt_local": initial_prompt,
            "generated_jailbreak": generated_jailbreak
        }

        # Check if local generation failed before proceeding
        if generated_jailbreak.startswith("[ERROR:"):
            logging.warning("Skipping OpenRouter evaluation for this sample due to local generation error.")
            # Add placeholder errors for API models in the results
            for model_id in OPENROUTER_MODELS:
                result_entry[f"response_{model_id}"] = "[SKIPPED DUE TO LOCAL ERROR]"
            results.append(result_entry)
            continue # Move to the next sample

        # --- Step 2: Feed the generated jailbreak to OpenRouter models ---
        for model_id in OPENROUTER_MODELS:
            logging.info(f"Feeding generated jailbreak to OpenRouter model ({model_id})...")
            # Use the 'generated_jailbreak' as the prompt for the API call
            api_output = generate_openrouter(
                openrouter_client, model_id, generated_jailbreak, MAX_NEW_TOKENS_API, TEMPERATURE, TOP_P
            )
            # Store the response under a distinct key
            result_entry[f"response_{model_id}"] = api_output
            logging.info(f"{model_id} Response to Jailbreak:\n{api_output}\n" + "-"*20)

        results.append(result_entry)

    # --- Display Results ---
    logging.info("\n\n--- Evaluation Summary ---")
    results_df = pd.DataFrame(results)

    # Define column order for clarity
    column_order = [
        "source_filename",
        "harmful_request_context",
        "initial_prompt_local",
        "generated_jailbreak"
    ] + [f"response_{model_id}" for model_id in OPENROUTER_MODELS]

    # Ensure only existing columns are selected (handles potential errors)
    results_df = results_df[[col for col in column_order if col in results_df.columns]]

    # Print DataFrame
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_colwidth', 200) # Adjust width as needed
    # print(results_df.to_markdown(index=False))

    # Optionally save to CSV
    results_df.to_csv("evaluation_results_piped_jailbreaks.csv", index=False)
    logging.info("Results saved to evaluation_results_piped_jailbreaks.csv")

    logging.info("\nEvaluation complete.")
