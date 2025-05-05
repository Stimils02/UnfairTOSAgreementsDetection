"""
Evaluation script for TinyLlama anomaly detection model.
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

# Load validation dataset
val_dataset = load_from_disk("./data/val")
print(f"Loaded validation dataset with {len(val_dataset)} examples")

# Load tokenizer
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load fine-tuned adapter - CORRECTED PATH
adapter_path = "./lora-tinyllama/checkpoint-30"  # Using the checkpoint directory
print(f"Loading adapter from: {adapter_path}")
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

# Function to generate predictions for a single example
def predict(text):
    prompt = f"<s>[CLAUSE]: {text} \n[Is this anomalous?]:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=5,
            temperature=0.1,
            do_sample=False
        )
    
    # Get the generated tokens (excluding the input)
    generated = outputs[0][len(inputs.input_ids[0]):]
    response = tokenizer.decode(generated, skip_special_tokens=True).strip()
    
    # Debug: print raw response
    # print(f"Raw response: '{response}'")
    
    # Simple yes/no extraction
    is_anomalous = 1 if "yes" in response.lower() else 0
    return response, is_anomalous

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
        print(f"  True label: {label} ({'Anomalous' if label == 1 else 'Not Anomalous'})")
        print(f"  Prediction: {pred} ({'Anomalous' if pred == 1 else 'Not Anomalous'})")
        print(f"  Full response: '{response}'")
        print()

# Calculate metrics
accuracy = accuracy_score(true_labels, predictions)
precision, recall, f1, _ = precision_recall_fscore_support(
    true_labels, predictions, average='binary', zero_division=0
)
cm = confusion_matrix(true_labels, predictions)

# Print class distribution
anomalous_preds = sum(predictions)
anomalous_true = sum(true_labels)
print(f"Prediction distribution: {anomalous_preds} anomalous, {len(predictions) - anomalous_preds} not anomalous")
print(f"True label distribution: {anomalous_true} anomalous, {len(true_labels) - anomalous_true} not anomalous")

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
results_df.to_csv('evaluation_results.csv', index=False)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Anomalous', 'Anomalous'],
            yticklabels=['Not Anomalous', 'Anomalous'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png')

print("\nResults saved to evaluation_results.csv")
print("Confusion matrix visualization saved to confusion_matrix.png")