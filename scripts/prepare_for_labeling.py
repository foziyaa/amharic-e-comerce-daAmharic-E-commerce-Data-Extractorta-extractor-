# scripts/prepare_for_labeling.py

import pandas as pd
from pathlib import Path

# --- Configuration ---
# Set up paths relative to the script's location
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Input file from Task 1
INPUT_CSV = PROJECT_ROOT / "dataraw" / "scraped_telegram_data.csv"

# Output file where we will save the messages to be labeled
OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "messages_for_labeling.csv"

# Number of messages we want to select
NUM_MESSAGES_TO_SELECT = 50

# --- Main Logic ---

def select_messages_for_labeling():
    """
    Reads the scraped data, selects the best messages for NER labeling,
    and saves them to a new CSV file.
    """
    print(f"Reading scraped data from: {INPUT_CSV}")
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"Error: Input file not found at {INPUT_CSV}")
        print("Please make sure you have run the scraper script first.")
        return

    # --- Find the best candidates for labeling ---
    # We want messages that are likely to contain our entities.
    # A good heuristic is to find messages that contain keywords for both PRICE and LOCATION.
    
    price_keywords = ['ብር', 'ዋጋ']
    location_keywords = ['አድራሻ', 'ቦሌ', 'ፒያሳ', 'መገናኛ', 'ካሳንቺስ', 'ሳሪስ'] # Add more common locations if you know them

    # Create boolean masks
    contains_price = df['message_text'].str.contains('|'.join(price_keywords), na=False)
    contains_location = df['message_text'].str.contains('|'.join(location_keywords), na=False)

    # Filter for messages that contain BOTH a price and a location keyword
    good_candidates = df[contains_price & contains_location]
    
    print(f"Found {len(good_candidates)} messages containing both price and location keywords.")

    # If we don't find enough good candidates, we'll fall back to just price
    if len(good_candidates) < NUM_MESSAGES_TO_SELECT:
        print("Not enough messages with both price and location. Falling back to messages with just a price keyword.")
        good_candidates = df[contains_price]

    # Make sure we don't try to select more messages than we have
    if len(good_candidates) == 0:
        print("Error: No messages found with relevant keywords. Cannot create labeling file.")
        return
        
    num_to_sample = min(NUM_MESSAGES_TO_SELECT, len(good_candidates))
    
    # Select a random sample of the best candidates
    messages_to_label_df = good_candidates.sample(n=num_to_sample, random_state=42) # random_state ensures we get the same sample every time

    # We only need the message_text column
    final_df = messages_to_label_df[['message_text']]

    # --- Save the result to a new CSV ---
    print(f"\nSaving {len(final_df)} selected messages to: {OUTPUT_CSV}")
    final_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    
    print("\nSuccessfully created the file for labeling.")
    print("You can now open this file and use it as a reference for Task 2.")


if __name__ == "__main__":
    select_messages_for_labeling()