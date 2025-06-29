# scripts/interpret_model.py

from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from pathlib import Path
import shap
import warnings
import numpy as np

# Suppress specific warnings that are not critical
warnings.filterwarnings("ignore", message=".*is not generative.*")
warnings.filterwarnings("ignore", message=".*internal specific backward pass.*")

# --- 1. Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "xlm-roberta-amharic-ner"

# --- 2. Load the Fine-Tuned Model and Components ---
print(f"Loading fine-tuned model from: {MODEL_DIR}")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
except OSError:
    print(f"FATAL ERROR: Model not found at {MODEL_DIR}. Please run training script first.")
    exit()

# --- THE FIX IS HERE ---
# We create the pipeline without the 'return_all_scores' argument,
# as your library version does not support it.
ner_pipeline = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="none" # This is the most important part for SHAP
)
print("NER pipeline created successfully.")


# --- 3. Use SHAP to Explain a Prediction ---
print("\nInitializing SHAP explainer...")
explainer = shap.Explainer(ner_pipeline)

amharic_post = "ዘመናዊ የልብስ ማስጫ ዋጋ 4000 ብር ሲሆን አድራሻችን መገናኛ ዘፍመሽ ሞል ነው"

print(f"\nAnalyzing text: '{amharic_post}'")
print("Generating SHAP values (this might take a moment)...")
shap_values = explainer([amharic_post])


# --- 4. Generate and Print a TEXT-BASED Summary ---
print("\n--- SHAP Analysis Summary (Text-Based) ---")

tokens = shap_values.data[0]

key_entities = ["B-PRODUCT", "B-PRICE", "B-LOC", "O"]

for entity in key_entities:
    print(f"\n--- Analysis for Entity: {entity} ---")
    try:
        entity_values = shap_values[0, :, entity].values
        
        word_value_pairs = list(zip(tokens, entity_values))
        word_value_pairs.sort(key=lambda x: x[1], reverse=True)
        
        print("  Top Positive Words (reasons TO predict this label):")
        for word, value in word_value_pairs[:3]:
            print(f'    - "{word}" (Influence: {value:.4f})')
            
        print("\n  Top Negative Words (reasons NOT to predict this label):")
        for word, value in reversed(word_value_pairs[-3:]):
            print(f'    - "{word}" (Influence: {value:.4f})')

    except Exception as e:
        print(f"  - Error generating text summary for '{entity}': {e}")

print("\n\nSUCCESS! Text-based analysis is complete.")
print("You can copy the summary above directly into your report.")