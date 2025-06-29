# scripts/split_data.py

import random
from pathlib import Path
import math

# --- Configuration ---
# Set up paths relative to the script's location
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Input file
INPUT_FILE = DATA_PROCESSED_DIR / "labeled_data.txt"

# Output files
# The script will overwrite the empty train.txt and validation.txt files
TRAIN_FILE = DATA_PROCESSED_DIR / "train.txt"
VALIDATION_FILE = DATA_PROCESSED_DIR / "validation.txt"
TEST_FILE = DATA_PROCESSED_DIR / "test.txt" # This will be created for completeness.

# Split ratios
TRAIN_RATIO = 0.8
VALIDATION_RATIO = 0.1
# Test ratio will be the remainder (0.1)

# For reproducibility of the split
RANDOM_SEED = 42

# --- Helper Functions ---

def read_conll_messages(file_path):
    """
    Reads a CoNLL formatted file and groups lines into messages.
    Messages are separated by blank lines.
    It ignores lines starting with '#' as comments.
    """
    if not file_path.exists():
        print(f"Error: Input file not found at {file_path}")
        return None

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    all_messages = []
    current_message = []
    for line in lines:
        line = line.strip()
        if line: # If the line is not empty
            if not line.startswith('#'): # Ignore comment lines
                current_message.append(line)
        else: # If the line is empty, it's a message separator
            if current_message:
                all_messages.append(current_message)
                current_message = []
    
    # Add the last message if the file doesn't end with a blank line
    if current_message:
        all_messages.append(current_message)
        
    return all_messages


def write_conll_messages(messages, output_path):
    """
    Writes a list of messages to a file in CoNLL format.
    Each message is separated by a blank line.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, message in enumerate(messages):
            for line in message:
                f.write(line + '\n')
            # Add a blank line separator, but not after the very last message
            if i < len(messages) - 1:
                f.write('\n')
    print(f"Successfully wrote {len(messages)} messages to {output_path}")


# --- Main Logic ---

def split_dataset():
    """
    Reads the full labeled dataset, shuffles it at the message level, 
    and splits it into training, validation, and test sets.
    """
    print(f"Reading data from: {INPUT_FILE}")
    messages = read_conll_messages(INPUT_FILE)

    if messages is None or not messages:
        print("No messages found or file could not be read. Halting script.")
        return

    print(f"Found a total of {len(messages)} messages to split.")
    
    # Shuffle the messages for random distribution
    random.seed(RANDOM_SEED)
    random.shuffle(messages)
    
    # Calculate split indices to ensure all data is used
    total_count = len(messages)
    train_count = math.ceil(total_count * TRAIN_RATIO)
    validation_count = math.floor(total_count * VALIDATION_RATIO)
    
    # Define the slice points in the shuffled list
    train_end_idx = train_count
    validation_end_idx = train_count + validation_count

    # Split the data into three lists
    train_messages = messages[:train_end_idx]
    validation_messages = messages[train_end_idx:validation_end_idx]
    test_messages = messages[validation_end_idx:]

    print("\n--- Split Summary ---")
    print(f"Training set:   {len(train_messages)} messages")
    print(f"Validation set: {len(validation_messages)} messages")
    print(f"Test set:       {len(test_messages)} messages")
    print("-----------------------\n")
    
    # Write the splits to their respective files
    write_conll_messages(train_messages, TRAIN_FILE)
    write_conll_messages(validation_messages, VALIDATION_FILE)
    write_conll_messages(test_messages, TEST_FILE)
    
    print("\nData splitting complete!")
    print(f"Files are located in: {DATA_PROCESSED_DIR}")

if __name__ == "__main__":
    split_dataset()