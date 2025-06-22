# scripts/scraper.py

import pandas as pd
import re
import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv
from telethon.sync import TelegramClient

# --- Secure Configuration ---

# Load environment variables from the .env file in the project's root directory
# This line looks for the .env file starting from where the script is run
project_root = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=project_root / '.env')

# Load keys safely from the environment. They are no longer in the code.
api_id = os.getenv("API_ID")
api_hash = os.getenv("API_HASH")

# Fail-fast: The script will stop if the keys are not found in your .env file.
if not api_id or not api_hash:
    print("FATAL ERROR: API_ID and/or API_HASH not found in your .env file.")
    print("Please create a .env file in the project root with your credentials.")
    exit() # Stop the script immediately

# --- Path and File Configuration ---
session_name = 'my_telegram_session'
DATA_RAW_DIR = project_root / "dataraw"

CHANNELS_FILE = DATA_RAW_DIR / "channels_to_crawl.csv"
OUTPUT_CSV_FILE = DATA_RAW_DIR / "scraped_telegram_data.csv"

# --- Helper Functions ---

def get_channels_from_csv(file_path):
    """Reads channel usernames from the specified CSV file."""
    try:
        df = pd.read_csv(file_path)
        if 'channel_username' not in df.columns:
            print(f"Error: The CSV file at {file_path} must have a column with the exact header 'channel_username'.")
            return []
        
        channels = df['channel_username'].dropna().tolist()
        print(f"Successfully loaded {len(channels)} channels from CSV file.")
        return channels
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found. Make sure it's in the 'dataraw' folder.")
        return []
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")
        return []

def clean_text(text):
    """Removes URLs and extra whitespace from text."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Main Scraping Logic ---

async def scrape_channels(channel_usernames):
    """Connects to Telegram and scrapes messages from the provided list of channels."""
    all_messages = []
    
    # Place the session file in the project root to keep the scripts folder clean
    session_path = str(project_root / session_name)
    async with TelegramClient(session_path, api_id, api_hash) as client:
        print("Client connected. Starting to scrape channels...")
        
        for channel_name in channel_usernames:
            if not isinstance(channel_name, str) or not channel_name.strip():
                continue

            print(f"Fetching messages from: {channel_name}...")
            try:
                channel_entity = await client.get_entity(channel_name)
                
                async for message in client.iter_messages(channel_entity, limit=500):
                    if message and message.text:
                        message_data = {
                            'channel_name': channel_name,
                            'timestamp': message.date,
                            'message_text': clean_text(message.text),
                            'views': message.views,
                            'has_image': message.photo is not None
                        }
                        all_messages.append(message_data)
                        
            except ValueError:
                 print(f"--> Could not find channel '{channel_name}'. Please check the username is correct and public.")
            except Exception as e:
                print(f"--> An unexpected error occurred with channel '{channel_name}'. Error: {e}")
    
    print(f"\nScraping complete. Found {len(all_messages)} total messages.")
    return all_messages

# --- Execution Block ---

if __name__ == "__main__":
    target_channels = get_channels_from_csv(CHANNELS_FILE)
    
    if target_channels:
        messages = asyncio.run(scrape_channels(target_channels))
        
        if messages:
            df = pd.DataFrame(messages)
            df.to_csv(OUTPUT_CSV_FILE, index=False, encoding='utf-8-sig')
            
            print(f"\nData successfully saved to {OUTPUT_CSV_FILE}")
            print("\n--- Data Sample ---")
            print(df.head())
        else:
            print("No new messages were scraped. The output file was not created.")
    else:
        print("Halting script because no channels were loaded from the CSV file.")