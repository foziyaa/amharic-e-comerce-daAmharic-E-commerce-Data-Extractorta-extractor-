# scripts/train_robust_model.py

import pandas as pd
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForTokenClassification
)
import numpy as np
from seqeval.metrics import classification_report

# --- 1. Configuration ---
MODEL_CHECKPOINT = "xlm-roberta-base"
MODEL_NAME = "xlm-roberta-amharic-ner"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "models" / MODEL_NAME

# --- 2. Load Labeled Data ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_FILE = PROJECT_ROOT / "dataprocessed" / "train.txt"
VALIDATION_FILE = PROJECT_ROOT / "dataprocessed" / "validation.txt"

def get_all_labels(files):
    unique_labels = set()
    for file in files:
        if not file.exists():
            print(f"FATAL ERROR: Data file not found at {file}. Please check the path.")
            exit()
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    parts = line.strip().split()
                    if len(parts) > 1:
                        unique_labels.add(parts[-1])
    unique_labels.add("O")
    return sorted(list(unique_labels))

all_labels = get_all_labels([TRAIN_FILE, VALIDATION_FILE])
label2id = {label: i for i, label in enumerate(all_labels)}
id2label = {i: label for i, label in enumerate(all_labels)}

print(f"Discovered Labels: {all_labels}")
print(f"Label to ID mapping: {label2id}")

def create_dataset_from_conll(file_path):
    tokens_list, tags_list = [], []
    current_tokens, current_tags = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == "" or line.startswith('#'):
                if current_tokens:
                    tokens_list.append(current_tokens)
                    tags_list.append(current_tags)
                    current_tokens, current_tags = [], []
            else:
                parts = line.split()
                if len(parts) == 2:
                    current_tokens.append(parts[0])
                    current_tags.append(parts[1] if parts[1] in label2id else "O")
    if current_tokens:
        tokens_list.append(current_tokens)
        tags_list.append(current_tags)
    tags_ids_list = [[label2id[tag] for tag in tags] for tags in tags_list]
    return Dataset.from_dict({"tokens": tokens_list, "ner_tags": tags_ids_list})

train_dataset = create_dataset_from_conll(TRAIN_FILE)
validation_dataset = create_dataset_from_conll(VALIDATION_FILE)

# --- 3. Tokenization & Label Alignment ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
tokenized_validation_dataset = validation_dataset.map(tokenize_and_align_labels, batched=True)

# --- 4. Model Training ---
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_CHECKPOINT, num_labels=len(all_labels), id2label=id2label, label2id=label2id
)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [[id2label[p] for (p, l) in zip(pred, lab) if l != -100] for pred, lab in zip(predictions, labels)]
    true_labels = [[id2label[l] for (p, l) in zip(pred, lab) if l != -100] for pred, lab in zip(predictions, labels)]
    report = classification_report(y_true=true_labels, y_pred=true_predictions, output_dict=True, zero_division=0)
    return {
        "precision": report["micro avg"]["precision"],
        "recall": report["micro avg"]["recall"],
        "f1-score": report["micro avg"]["f1-score"],
    }

# --- IMPROVED TRAINING ARGUMENTS ---
training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    log_level="error",
    num_train_epochs=25,  # Increased epochs for more learning
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=3e-5, # Slightly adjusted learning rate
    weight_decay=0.01,
    # Evaluation and saving strategy
    eval_strategy="epoch",
    save_strategy="epoch",
    # Technical improvements
    fp16=False, # Set to True if you have a modern NVIDIA GPU
    # Logging
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1-score",
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_validation_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("\n--- Starting ROBUST Model Training ---")
trainer.train()
print("--- Training Complete ---")

trainer.save_model(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))
print(f"\nModel saved to {OUTPUT_DIR}")

# --- Final Evaluation ---
print("\n--- Final Model Performance ---")
final_metrics = trainer.evaluate()
print(final_metrics)