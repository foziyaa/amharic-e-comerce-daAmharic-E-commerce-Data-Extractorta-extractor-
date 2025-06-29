import pandas as pd
from pathlib import Path
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import argparse
import numpy as np
import re

# --- 1. Configuration & Argument Parsing ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "dataraw"
INPUT_CSV = DATA_RAW_DIR / "scraped_telegram_data.csv"

parser = argparse.ArgumentParser(description="Generate a FinTech Vendor Scorecard using an NER model.")
parser.add_argument(
    "--model_name",
    type=str,
    required=True,
    help="Name of the fine-tuned model folder inside the 'models' directory (e.g., 'xlm-roberta-ner')."
)
args = parser.parse_args()

MODEL_PATH = PROJECT_ROOT / "models" / args.model_name

# --- 2. Helper Function to Extract Price from NER Results ---
# --- 2. Helper Function to Extract Price from NER Results ---
def extract_price_from_entities(entities):
    """
    Parses NER pipeline output (with aggregation_strategy='simple') to find and convert a price.
    """
    # The 'entities' variable is a list of dictionaries, e.g.,
    # [{'entity_group': 'PRICE', 'word': '3500 ብር', ...}]
    
    for entity in entities:
        # We now look for the 'entity_group' key and the 'PRICE' value
        if entity['entity_group'] == 'PRICE':
            # The 'word' key already contains the full phrase, e.g., "3500 ብር"
            price_string = entity['word']
            
            # Keep only digits and decimal points
            numeric_price_string = re.sub(r'[^\d.]', '', price_string)
            
            if numeric_price_string:
                try:
                    return float(numeric_price_string)
                except (ValueError, TypeError):
                    # If conversion fails, continue to check if there's another PRICE entity
                    continue
    
    # If no PRICE entity was found in the list for this message
    return np.nan
# --- 3. Main Logic: The Vendor Analytics Engine ---
def create_scorecard():
    """
    Loads scraped data, uses NER to extract prices, calculates vendor metrics,
    and generates a final scorecard.
    """
    # --- Load Model ---
    print(f"--- Step 1: Loading NER model '{args.model_name}' ---")
    if not MODEL_PATH.exists():
        print(f"FATAL ERROR: Model not found at {MODEL_PATH}. Please run training first.")
        return
    
    ner_pipeline = pipeline(
        "ner",
        model=str(MODEL_PATH),
        tokenizer=str(MODEL_PATH),
        aggregation_strategy="simple" # Groups subwords, e.g., 'መኪና' -> 'መኪና'
    )

    # --- Load Data ---
    print(f"--- Step 2: Loading scraped data from {INPUT_CSV} ---")
    if not INPUT_CSV.exists():
        print(f"FATAL ERROR: Scraped data not found at {INPUT_CSV}. Please run the scraper first.")
        return
        
    df = pd.read_csv(INPUT_CSV)
    df.dropna(subset=['message_text'], inplace=True) # Ensure all rows have text
    
    # --- Extract Prices using NER ---
    print("--- Step 3: Extracting prices from messages (this may take a while)... ---")
    all_texts = df['message_text'].tolist()
    ner_results = ner_pipeline(all_texts)
    
    df['extracted_price'] = [extract_price_from_entities(result) for result in ner_results]

    # --- Calculate Key Vendor Metrics ---
    print("--- Step 4: Calculating vendor performance metrics... ---")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Group by vendor (channel) to calculate stats
    vendor_metrics = df.groupby('channel_name').agg(
        Post_Count=('timestamp', 'size'),
        Avg_Views_Per_Post=('views', 'mean'),
        Avg_Product_Price_ETB=('extracted_price', 'mean'),
        First_Post_Date=('timestamp', 'min'),
        Last_Post_Date=('timestamp', 'max')
    ).reset_index()

    # Calculate Posting Frequency (Posts per Week)
    vendor_metrics['Observed_Period_Days'] = (vendor_metrics['Last_Post_Date'] - vendor_metrics['First_Post_Date']).dt.days + 1
    vendor_metrics['Posts_Per_Week'] = (vendor_metrics['Post_Count'] / vendor_metrics['Observed_Period_Days']) * 7
    
    # Clean up the final metrics
    vendor_metrics = vendor_metrics.round(2)

    # --- Create the Final Lending Score ---
    print("--- Step 5: Generating the final Lending Score... ---")
    
    # Normalize metrics to be on a similar scale (0-1) to prevent one metric from dominating
    for col in ['Avg_Views_Per_Post', 'Posts_Per_Week', 'Avg_Product_Price_ETB']:
        min_val = vendor_metrics[col].min()
        max_val = vendor_metrics[col].max()
        if (max_val - min_val) > 0:
            vendor_metrics[f'norm_{col}'] = (vendor_metrics[col] - min_val) / (max_val - min_val)
        else:
            vendor_metrics[f'norm_{col}'] = 0 # Handle case where all values are the same
            
    # Define the weights for our score. More weight on views (reach) and frequency (activity).
    # Reach (Views) = 50%, Activity (Frequency) = 40%, Scale (Price) = 10%
    vendor_metrics['Lending_Score'] = (
        vendor_metrics['norm_Avg_Views_Per_Post'] * 0.50 +
        vendor_metrics['norm_Posts_Per_Week'] * 0.40 +
        vendor_metrics['norm_Avg_Product_Price_ETB'] * 0.10
    ) * 100 # Scale to 100 for readability

    # --- Display the Final Scorecard ---
    final_scorecard = vendor_metrics[[
        'channel_name',
        'Lending_Score',
        'Avg_Views_Per_Post',
        'Posts_Per_Week',
        'Avg_Product_Price_ETB',
        'Post_Count'
    ]].sort_values(by='Lending_Score', ascending=False).reset_index(drop=True)

    print("\n\n" + "="*80)
    print(" " * 25 + "FINTECH VENDOR SCORECARD")
    print("="*80)
    print("Vendors are ranked by a 'Lending Score' based on market reach, activity, and scale.")
    print("Higher scores indicate stronger candidates for micro-lending.\n")
    print(final_scorecard.to_string())
    print("="*80)


if __name__ == "__main__":
    create_scorecard()