"""
Fine-tuning script for Equall/Saul-7B-Base on anomaly detection in legal clauses.
"""
import torch
import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import transformers
from transformers import Trainer
import os

# Check versions
print("Transformers version:", transformers.__version__)
from transformers import TrainingArguments
print("TrainingArguments path:", TrainingArguments.__module__)

# Load dataset
train_dataset = load_from_disk("./data/train")
val_dataset = load_from_disk("./data/val")

print(f"Training dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")

# Prompt formatting
def format_prompt(example):
    prompt = f"<s>[INST] You are a legal expert specializing in consumer protection. Review the following clause from a terms of service agreement:\n\n\"{example['text']}\"\n\nIs this clause anomalous or unfair to consumers? Answer with Yes or No and explain why. [/INST]"
    label = " Yes, this clause is anomalous because it unfairly restricts consumer rights." if example["label"] == 1 else " No, this clause is standard and not unfair to consumers."
    return {"text": prompt + label, "label": example["label"]}

train_dataset = train_dataset.map(format_prompt)
val_dataset = val_dataset.map(format_prompt)

# Load tokenizer
model_name = "Equall/Saul-7B-Base"
print(f"Loading tokenizer from {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Quant config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Load model
print(f"Loading Saul-7B-Base model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Apply LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ✅ Fixed tokenizer mapping
def tokenize(example):
    tokens = tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    tokens = {k: v.squeeze(0) for k, v in tokens.items()}
    tokens["labels"] = tokens["input_ids"].clone()
    return tokens

train_dataset = train_dataset.map(tokenize, remove_columns=train_dataset.column_names)
val_dataset = val_dataset.map(tokenize, remove_columns=val_dataset.column_names)

# Metrics
def compute_metrics(eval_preds):
    predictions = eval_preds.predictions
    labels = eval_preds.label_ids

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    pred_labels = []
    true_labels = []

    print(f"Number of samples in evaluation: {len(predictions)}")

    for i, (pred, label) in enumerate(zip(predictions, labels)):
        valid_indices = np.where(label != -100)[0]
        if len(valid_indices) == 0:
            continue
        last_idx = valid_indices[-1]
        pred_token_id = np.argmax(pred[last_idx]) if len(pred.shape) > 1 else np.argmax(pred)
        label_token_id = label[last_idx]
        pred_text = tokenizer.decode([pred_token_id]).lower()
        label_text = tokenizer.decode([label_token_id]).lower()
        if i < 5:
            print(f"Example {i}:\n  Pred token: {pred_token_id}, text: '{pred_text}'\n  Label token: {label_token_id}, text: '{label_text}'")
        is_anomalous_pred = 1 if "yes" in pred_text else 0
        is_anomalous_true = 1 if "yes" in label_text else 0
        pred_labels.append(is_anomalous_pred)
        true_labels.append(is_anomalous_true)

    print(f"Predicted class distribution: 0s={pred_labels.count(0)}, 1s={pred_labels.count(1)}")
    print(f"True class distribution: 0s={true_labels.count(0)}, 1s={true_labels.count(1)}")

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average="binary", zero_division=0
    )
    acc = accuracy_score(true_labels, pred_labels)

    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# Custom Trainer with fixed compute_loss method
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss if hasattr(outputs, "loss") else None
        return (loss, outputs) if return_outputs else loss

output_dir = "./saul-7b-anomaly"
os.makedirs(output_dir, exist_ok=True)

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=2,
    learning_rate=1e-4,
    eval_steps=20,
    save_steps=20,
    logging_steps=5,
    report_to="none",
    fp16=True,
    gradient_checkpointing=True,
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

print("Starting training...")
trainer.train()

print("Saving model...")
trainer.save_model(f"{output_dir}/final")
model.save_pretrained(f"{output_dir}/final")

# Final evaluation
print("Running final evaluation...")
outputs = trainer.predict(val_dataset)
preds = outputs.predictions
labels = outputs.label_ids

def extract_predictions(predictions, labels, tokenizer):
    pred_labels, true_labels = [], []
    for i, (pred, label) in enumerate(zip(predictions, labels)):
        if isinstance(pred, tuple): pred = pred[0]
        valid_indices = np.where(label != -100)[0]
        if len(valid_indices) == 0: continue
        last_idx = valid_indices[-1]
        pred_token = np.argmax(pred[last_idx]) if len(pred.shape) > 1 else np.argmax(pred)
        label_token = label[last_idx]
        pred_text = tokenizer.decode([pred_token]).lower()
        label_text = tokenizer.decode([label_token]).lower()
        pred_binary = 1 if "yes" in pred_text else 0
        true_binary = 1 if "yes" in label_text else 0
        pred_labels.append(pred_binary)
        true_labels.append(true_binary)
    return pred_labels, true_labels

with open(f"{output_dir}/predictions.txt", "w") as f:
    pred_binary, true_binary = extract_predictions(preds, labels, tokenizer)
    for i, (pred, true) in enumerate(zip(pred_binary, true_binary)):
        pred_str = "Yes, anomalous" if pred == 1 else "No, not anomalous"
        true_str = "Yes, anomalous" if true == 1 else "No, not anomalous"
        f.write(f"Example {i}: Prediction: {pred_str} | True: {true_str}\n")

accuracy = accuracy_score(true_binary, pred_binary)
precision, recall, f1, _ = precision_recall_fscore_support(
    true_binary, pred_binary, average="binary", zero_division=0
)

print("\n=== FINAL EVALUATION RESULTS ===")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

from sklearn.metrics import classification_report, confusion_matrix
print("\nDetailed Classification Report:")
print(classification_report(true_binary, pred_binary, zero_division=0))
print("\nConfusion Matrix:")
print(confusion_matrix(true_binary, pred_binary))
print("\nTraining and evaluation complete!")