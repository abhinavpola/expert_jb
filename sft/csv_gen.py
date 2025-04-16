import csv
import re
import os
from pathlib import Path

def build_csv_from_markdown(input_dir: str, output_csv: str):
    """
    Finds all .mkd files in input_dir, extracts content under each Markdown header,
    and writes the results to output_csv.

    Args:
        input_dir: The directory containing the .mkd files (searches recursively).
        output_csv: The path to the output CSV file.
    """
    data_rows = []
    # Regex to find Markdown headers (lines starting with #)
    # ^      - Start of the line (with re.MULTILINE)
    # #+     - One or more '#' characters
    # \s+    - One or more whitespace characters
    # (.*)   - Capture group for the header text itself
    header_pattern = re.compile(r"^#+\s+(.*)", re.MULTILINE)

    input_path = Path(input_dir)
    if not input_path.is_dir():
        print(f"Error: Input directory '{input_dir}' not found.")
        return

    print(f"Searching for .mkd files in '{input_dir}'...")
    mkd_files = list(input_path.rglob('*.mkd')) # rglob searches recursively

    if not mkd_files:
        print("No .mkd files found.")
        return

    print(f"Found {len(mkd_files)} .mkd files. Processing...")

    for file_path in mkd_files:
        print(f"  Processing: {file_path}")
        try:
            content = file_path.read_text(encoding='utf-8')
            # Find all header matches with their positions
            matches = list(header_pattern.finditer(content))

            if not matches:
                # Option 1: Add the whole file content with a null/placeholder header
                # data_rows.append({
                #     'filename': str(file_path.relative_to(input_path)),
                #     'header': None, # Or some placeholder like "[NO HEADER]"
                #     'content': content.strip()
                # })
                # Option 2: Skip files without headers (current implementation)
                print(f"    Skipping {file_path} - No headers found.")
                continue

            # Iterate through headers to extract content between them
            for i, match in enumerate(matches):
                header_text = match.group(1).strip()
                # Start position of content is right after the header line
                content_start = match.end()
                # End position of content is right before the next header starts,
                # or the end of the file if it's the last header
                content_end = matches[i+1].start() if (i + 1) < len(matches) else len(content)

                # Extract the content block and strip leading/trailing whitespace
                section_content = content[content_start:content_end].strip()

                # Only add row if there is actual content under the header
                if section_content:
                    data_rows.append({
                        'filename': str(file_path.relative_to(input_path)), # Store relative path
                        'header': header_text,
                        'content': section_content
                    })

        except FileNotFoundError:
            print(f"    Error: File not found (should not happen with Path.rglob): {file_path}")
        except Exception as e:
            print(f"    Error processing file {file_path}: {e}")

    if not data_rows:
        print("No data extracted from files.")
        return

    print(f"\nWriting {len(data_rows)} rows to '{output_csv}'...")
    try:
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['filename', 'header', 'content']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerows(data_rows)
        print("CSV file created successfully.")
    except IOError as e:
        print(f"Error writing CSV file '{output_csv}': {e}")
    except Exception as e:
        print(f"An unexpected error occurred during CSV writing: {e}")


# --- Configuration ---
INPUT_DIRECTORY = "L1B3RT4S"  # <--- CHANGE THIS
OUTPUT_CSV_FILE = "output_markdown_data.csv"   # <--- CHANGE THIS (optional)
# -------------------

if __name__ == "__main__":
    # Basic check if the user likely forgot to change the input directory
    if INPUT_DIRECTORY == "path/to/your/markdown/files":
         print("Please update the INPUT_DIRECTORY variable in the script before running.")
    else:
        build_csv_from_markdown(INPUT_DIRECTORY, OUTPUT_CSV_FILE)
