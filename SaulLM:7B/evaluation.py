"""
Evaluation script for SaulLM-7B fine-tuned for legal Fairness detection.
"""
import torch
import numpy as np
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Load validation dataset
val_dataset = load_from_disk("./data/val")
print(f"Loaded validation dataset with {len(val_dataset)} examples")

# Load tokenizer
model_name = "pierre-colombo/SaulLM-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load base model with 4-bit quantization to fit in memory
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

print(f"Loading base SaulLM-7B model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load fine-tuned adapter
adapter_path = "./saullm-lora-anomaly/final"  # Change this to your adapter path
print(f"Loading adapter from: {adapter_path}")

# Check if adapter path exists
if not os.path.exists(adapter_path):
    print(f"Warning: Adapter path {adapter_path} does not exist!")
    # Try to find the latest checkpoint
    checkpoints = [d for d in os.listdir("./saullm-lora-anomaly") if d.startswith("checkpoint")]
    if checkpoints:
        latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
        adapter_path = f"./saullm-lora-anomaly/{latest_checkpoint}"
        print(f"Using latest checkpoint instead: {adapter_path}")
    else:
        print("No checkpoints found! Proceeding with base model only.")

# Try to load the adapter if it exists
try:
    model = PeftModel.from_pretrained(model, adapter_path)
    print("Successfully loaded LoRA adapter")
except Exception as e:
    print(f"Error loading adapter: {e}")
    print("Proceeding with base model only")

model.eval()

# Function to generate predictions for a single example
def predict(text):
    # Using a legal-oriented prompt
    prompt = f"<s>[LEGAL CLAUSE]: {text} \n[QUESTION]: Is this clause Unfair or potentially unfair to consumers? \n[ANSWER]:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=20,  # Allow longer response for legal reasoning
            temperature=0.1,
            do_sample=False
        )
    
    # Get the generated tokens (excluding the input)
    generated = outputs[0][len(inputs.input_ids[0]):]
    response = tokenizer.decode(generated, skip_special_tokens=True).strip()
    
    # Extract yes/no from response
    is_Unfair = 1 if "yes" in response.lower() or "Unfair" in response.lower() else 0
    return response, is_Unfair

# Evaluate all examples
predictions = []
true_labels = []
full_responses = []

print("Evaluating examples...")
for i, example in enumerate(val_dataset):
    text = example["text"]
    label = example["label"]
    
    response, pred = predict(text)
    
    predictions.append(pred)
    true_labels.append(label)
    full_responses.append(response)
    
    if i < 10:  # Print first 10 examples for inspection
        print(f"Example {i}:")
        print(f"  Text: {text[:100]}...")
        print(f"  True label: {label} ({'Unfair' if label == 1 else 'Fair'})")
        print(f"  Prediction: {pred} ({'Unfair' if pred == 1 else 'Fair'})")
        print(f"  Full response: '{response}'")
        print()
    
    # Print progress every 10 examples
    if (i+1) % 10 == 0:
        print(f"Processed {i+1}/{len(val_dataset)} examples")

# Calculate metrics
accuracy = accuracy_score(true_labels, predictions)
precision, recall, f1, _ = precision_recall_fscore_support(
    true_labels, predictions, average='binary', zero_division=0
)
cm = confusion_matrix(true_labels, predictions)

# Print class distribution
Unfair_preds = sum(predictions)
Unfair_true = sum(true_labels)
print(f"Prediction distribution: {Unfair_preds} Unfair, {len(predictions) - Unfair_preds} Fair")
print(f"True label distribution: {Unfair_true} Unfair, {len(true_labels) - Unfair_true} Fair")

# Print results
print("\n=== EVALUATION RESULTS ===")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nDetailed Classification Report:")
print(classification_report(true_labels, predictions, zero_division=0))

# Calculate class distribution
true_positives = sum(1 for t, p in zip(true_labels, predictions) if t == 1 and p == 1)
false_positives = sum(1 for t, p in zip(true_labels, predictions) if t == 0 and p == 1)
true_negatives = sum(1 for t, p in zip(true_labels, predictions) if t == 0 and p == 0)
false_negatives = sum(1 for t, p in zip(true_labels, predictions) if t == 1 and p == 0)

print("\nPrediction Analysis:")
print(f"True Positives: {true_positives}")
print(f"False Positives: {false_positives}")
print(f"True Negatives: {true_negatives}")
print(f"False Negatives: {false_negatives}")

# Save detailed results to CSV
results_df = pd.DataFrame({
    'text': [ex['text'] for ex in val_dataset],
    'true_label': true_labels,
    'predicted_label': predictions,
    'response': full_responses,
    'correct': [t == p for t, p in zip(true_labels, predictions)]
})
results_df.to_csv('saullm_evaluation_results.csv', index=False)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Fair', 'Unfair'],
            yticklabels=['Fair', 'Unfair'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('saullm_confusion_matrix.png')

# Create additional visualizations
# 1. Examples by correctness
plt.figure(figsize=(10, 6))
correctness = ["Incorrect" if t != p else "Correct" for t, p in zip(true_labels, predictions)]
sns.countplot(x=correctness)
plt.title('Model Performance by Correctness')
plt.savefig('saullm_correctness.png')

# 2. Response length analysis
response_lengths = [len(resp) for resp in full_responses]
correct = [t == p for t, p in zip(true_labels, predictions)]

plt.figure(figsize=(10, 6))
sns.histplot(x=response_lengths, hue=correct, bins=20, kde=True)
plt.title('Response Length Distribution by Correctness')
plt.xlabel('Response Length (characters)')
plt.savefig('saullm_response_length.png')

print("\nResults saved to saullm_evaluation_results.csv")
print("Visualizations saved to saullm_confusion_matrix.png, saullm_correctness.png, and saullm_response_length.png")