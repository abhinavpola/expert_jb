import json
import os
import glob
import csv


def extract_compliant_examples():
    """
    1. Go through the results directory
    2. Read each model's final results JSON file
    3. Extract all entries with class_index 6 (Direct compliance)
    4. Add a 'model' field to each entry
    5. Include reasoning from prompts file
    6. Store all entries in a CSV file
    """
    # List to store all compliant examples
    compliant_examples = []

    # Load prompts file with reasoning
    prompts_file = "results/jailbreak_prompts_deepseek_r1.json"
    prompt_map = {}
    try:
        with open(prompts_file, "r") as f:
            prompts_data = json.load(f)

        # Create a mapping from prompt text to reasoning
        for prompt in prompts_data:
            prompt_text = prompt.get("prompt", "")
            reasoning = prompt.get("reasoning", "")
            prompt_map[prompt_text] = reasoning

        print(f"Loaded {len(prompt_map)} prompts with reasoning")
    except Exception as e:
        print(f"Error loading prompts file: {e}")
        prompt_map = {}

    # Find all provider directories in results
    providers = glob.glob("results/*")

    for provider_dir in providers:
        if not os.path.isdir(provider_dir):
            continue

        provider = os.path.basename(provider_dir)

        # Find all final result files for this provider
        final_files = glob.glob(f"{provider_dir}/*_final.json")

        for file_path in final_files:
            try:
                # Extract model name from filename
                filename = os.path.basename(file_path)
                model_name = filename.split("_")[0]
                full_model_name = f"{provider}/{model_name}"

                # Read the JSON file
                with open(file_path, "r") as f:
                    data = json.load(f)

                # Extract entries with class_index 6
                for entry in data:
                    if entry.get("class_index") == 6:
                        # Create a copy of the entry and add the model field
                        entry_with_model = entry.copy()
                        entry_with_model["model"] = full_model_name

                        # Add reasoning if available
                        prompt_text = entry.get("prompt", "")
                        if prompt_text in prompt_map:
                            entry_with_model["reasoning"] = prompt_map[prompt_text]
                        else:
                            entry_with_model["reasoning"] = ""

                        compliant_examples.append(entry_with_model)

                print(f"Processed: {full_model_name}")

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    # Write to CSV
    if compliant_examples:
        output_path = "results/compliant_examples.csv"

        # Get all unique field names from the data
        fieldnames = set()
        for example in compliant_examples:
            fieldnames.update(example.keys())
        fieldnames = sorted(list(fieldnames))

        # Make sure 'model' is the first column
        if "model" in fieldnames:
            fieldnames.remove("model")
            fieldnames = ["model"] + fieldnames

        with open(output_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(compliant_examples)

        print(f"\nExtracted {len(compliant_examples)} compliant examples")
        print(f"CSV file saved to: {output_path}")
    else:
        print("No compliant examples found.")


if __name__ == "__main__":
    extract_compliant_examples()
