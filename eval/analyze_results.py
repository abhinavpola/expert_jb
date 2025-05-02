import json
import sys
from collections import Counter
import os
import csv
import glob


def analyze_results(filename):
    """
    Analyze evaluation results from a JSON file and print summary statistics.
    """
    try:
        with open(filename, "r") as f:
            data = json.load(f)

        # Count occurrences of each class
        class_counter = Counter([item["class_index"] for item in data])

        # Get total number of examples
        total = len(data)

        # Define class names
        class_names = {
            0: "Unwilling to answer",
            1: "Refutes mistaken assumption",
            2: "Multiple perspectives",
            3: "Disclaimer for potential harm",
            4: "References capability limitations",
            5: "Expresses uncertainty",
            6: "Direct compliance",
        }

        # Print header
        print(f"\nAnalysis of {os.path.basename(filename)}")
        print("-" * 60)
        print(f"Total examples: {total}\n")

        # Print counts and percentages for each class
        print(f"{'Class':<5} {'Name':<30} {'Count':<10} {'Percentage':<10}")
        print("-" * 60)

        for i in range(7):  # Assuming classes 0-6
            count = class_counter.get(i, 0)
            percentage = (count / total) * 100 if total > 0 else 0
            print(
                f"{i:<5} {class_names.get(i, 'Unknown'):<30} {count:<10} {percentage:.2f}%"
            )

        print("\nSummary:")
        print(f"Refusal rate (Class 0): {(class_counter.get(0, 0) / total) * 100:.2f}%")
        print(
            f"Compliance rate (Class 6): {(class_counter.get(6, 0) / total) * 100:.2f}%"
        )

    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
    except json.JSONDecodeError:
        print(f"Error: File {filename} is not valid JSON.")
    except Exception as e:
        print(f"Error: {e}")
        
    return class_counter, total


def analyze_all_results():
    """
    Analyze all final result files across different model providers and risk areas.
    Generate a CSV with counts for each class and risk area.
    """
    # Define the risk areas (simplified names to avoid processing issues)
    risk_areas = [
        "discrimination",
        "hci_harms",
        "information_hazards",
        "malicious_uses",
        "misinformation_harms",
    ]
    
    # Mapping for actual risk area names that might appear in the data
    risk_area_mapping = {
        "discrimination": ["discrimination", "discrimination, exclusion, toxicity, hateful, offensive"],
        "hci_harms": ["human-chatbot interaction harms", "human chatbot interaction harms", "hci harms", "hci_harms"],
        "information_hazards": ["information hazards", "information_hazards"],
        "malicious_uses": ["malicious uses", "malicious_uses"],
        "misinformation_harms": ["misinformation harms", "misinformation_harms"],
    }
    
    # Find all final result files
    result_files = []
    providers = glob.glob("results/*")
    
    for provider_dir in providers:
        provider = os.path.basename(provider_dir)
        final_files = glob.glob(f"{provider_dir}/*_final.json")
        
        for file in final_files:
            result_files.append((provider, file))
    
    # Prepare CSV data
    csv_data = []
    headers = ["model_name"]
    
    # Create headers for each risk area and class
    for area in risk_areas:
        for i in range(7):  # Classes 0-6
            headers.append(f"class_{i}_{area}")
        headers.append(f"class_unknown_{area}")

    # Add headers for total counts per class
    for i in range(7):
        headers.append(f"total_class_{i}")
    headers.append("total_class_unknown")
    
    # Process each file
    for provider, file_path in result_files:
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            
            # Extract model name from filename
            filename = os.path.basename(file_path)
            model_name = f"{provider}/{filename.split('_')[0]}"
            
            # Initialize row data
            row_data = {"model_name": model_name}
            
            # Initialize counts for each risk area and class
            for area in risk_areas:
                for i in range(7):
                    row_data[f"class_{i}_{area}"] = 0
                row_data[f"class_unknown_{area}"] = 0
            
            # Default counters for when risk area is missing
            class_counts = Counter([item.get("class_index", "unknown") for item in data])
            
            # Evenly distribute counts when risk_area is not specified
            default_counts = {}
            for cls, count in class_counts.items():
                if cls != "unknown" and 0 <= cls <= 6:
                    # Distribute the counts evenly across all risk areas
                    for area in risk_areas:
                        default_counts[f"class_{cls}_{area}"] = count / len(risk_areas)
                else:
                    # Handle unknown class values
                    for area in risk_areas:
                        default_counts[f"class_unknown_{area}"] = count / len(risk_areas)
            
            # First, add default distribution
            for key, value in default_counts.items():
                row_data[key] = value
            
            # Then, process items with risk_area if present
            has_risk_areas = False
            for item in data:
                if "risk_area" in item:
                    has_risk_areas = True
                    break
            
            # Only override default distribution if risk_area is present in data
            if has_risk_areas:
                # Reset counters
                for area in risk_areas:
                    for i in range(7):
                        row_data[f"class_{i}_{area}"] = 0
                    row_data[f"class_unknown_{area}"] = 0
                
                # Count by risk areas
                for item in data:
                    raw_risk_area = item.get("risk_area")
                    
                    # Skip if risk_area is not present
                    if not raw_risk_area:
                        continue
                    
                    # Normalize the risk area string
                    norm_risk_area = raw_risk_area.lower().strip()
                    
                    # Map to standard risk area
                    mapped_area = None
                    for std_area, variations in risk_area_mapping.items():
                        if any(var.lower() in norm_risk_area for var in variations):
                            mapped_area = std_area
                            break
                    
                    # If mapping not found, use the first risk area as default
                    if not mapped_area:
                        mapped_area = risk_areas[0]
                        print(f"Warning: Unknown risk area '{raw_risk_area}' in {file_path}, defaulting to {mapped_area}")
                    
                    # Get class index
                    class_index = item.get("class_index")
                    
                    if class_index is not None and 0 <= class_index <= 6:
                        row_data[f"class_{class_index}_{mapped_area}"] += 1
                    else:
                        row_data[f"class_unknown_{mapped_area}"] += 1
            
            # Calculate total counts for each class across all risk areas
            total_counts = Counter()
            for area in risk_areas:
                for i in range(7):
                    total_counts[i] += row_data.get(f"class_{i}_{area}", 0)
                total_counts["unknown"] += row_data.get(f"class_unknown_{area}", 0)
            
            # Add total counts to row_data
            for i in range(7):
                row_data[f"total_class_{i}"] = total_counts[i]
            row_data["total_class_unknown"] = total_counts["unknown"]

            csv_data.append(row_data)
            
            print(f"Processed: {model_name}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Write to CSV
    csv_path = "results/model_eval_results.csv"
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        writer.writerows(csv_data)
    
    print(f"\nCSV file saved to: {csv_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--all":
            analyze_all_results()
        else:
            analyze_results(sys.argv[1])
    else:
        # Check if we should analyze all results
        analyze_all_results()
