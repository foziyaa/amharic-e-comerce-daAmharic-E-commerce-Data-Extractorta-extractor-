# scripts/extract_messages.py

import pandas as pd
from pathlib import Path

# --- Path Setup ---
# This ensures our script can find the files correctly
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Input file (the big one with all data)
INPUT_CSV = PROJECT_ROOT / "dataraw" / "scraped_telegram_data.csv"

# Output file (the new, simple one with just messages)
OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "all_messages_for_labeling.csv"

# --- Main Logic ---
def extract_message_column():
    """
    Reads the main scraped data CSV, extracts the 'message_text' column,
    and saves it to a new CSV file for easier review.
    """
    print(f"Reading data from: {INPUT_CSV}")
    try:
        # Read the full dataset
        df = pd.read_csv(INPUT_CSV)

        # Check if the 'message_text' column exists
        if 'message_text' not in df.columns:
            print("Error: 'message_text' column not found in the input file.")
            return

        # Create a new DataFrame with only the message column
        messages_df = df[['message_text']].copy()

        # Optional: Drop any rows where the message is empty or missing
        messages_df.dropna(inplace=True)

        # Save the new DataFrame to the processed data folder
        messages_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
        
        print("-" * 50)
        print("Success!")
        print(f"A new file has been created with all the messages.")
        print(f"File location: {OUTPUT_CSV}")
        print(f"Total messages extracted: {len(messages_df)}")
        print("-" * 50)

    except FileNotFoundError:
        print(f"Error: The input file was not found at {INPUT_CSV}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# --- Execution Block ---
if __name__ == "__main__":
    extract_message_column()