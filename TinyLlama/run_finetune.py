import torch
import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import transformers
# ✅ Check transformers version at runtime
print("Transformers version:", transformers.__version__)
from transformers import TrainingArguments
print("TrainingArguments path:", TrainingArguments.__module__)

# Load dataset
train_dataset = load_from_disk("/home/njuttu_umass_edu/685/ZeroShotAnomolyDetection/TinyLlama/data/train")
val_dataset = load_from_disk("/home/njuttu_umass_edu/685/ZeroShotAnomolyDetection/TinyLlama/data/val")

# Format prompt
def format_prompt(example):
    prompt = f"<s>[CLAUSE]: {example['text']} \n[Is this anomalous?]:"
    label = " Yes" if example["label"] == 1 else " No"
    return {"text": prompt + label, "label": example["label"]}

train_dataset = train_dataset.map(format_prompt)
val_dataset = val_dataset.map(format_prompt)

# Load tokenizer
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# ✅ Set quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Load model with 4-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Apply LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)

# Tokenize
def tokenize(example):
    inputs = tokenizer(example["text"], padding="max_length", truncation=True, max_length=256)
    inputs["labels"] = inputs["input_ids"]
    return inputs

train_dataset = train_dataset.map(tokenize, remove_columns=train_dataset.column_names)
val_dataset = val_dataset.map(tokenize, remove_columns=val_dataset.column_names)

# Compute metrics
def compute_metrics(eval_preds):
    predictions = np.argmax(eval_preds.predictions, axis=-1)
    labels = eval_preds.label_ids

    pred_labels = []
    true_labels = []
    for pred, label in zip(predictions, labels):
        pred_token = pred[-1]
        label_token = label[-1]
        pred_text = tokenizer.decode([pred_token])
        label_text = tokenizer.decode([label_token])

        pred_labels.append(1 if "yes" in pred_text.lower() else 0)
        true_labels.append(1 if "yes" in label_text.lower() else 0)

    acc = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average="binary")
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# Training arguments
training_args = TrainingArguments(
    output_dir="./lora-tinyllama",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    report_to="none",
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train
trainer.train()

# Evaluate and save predictions
outputs = trainer.predict(val_dataset)
preds = outputs.predictions
labels = outputs.label_ids

with open("predictions.txt", "w") as f:
    for i in range(len(labels)):
        pred_token = np.argmax(preds[i])
        pred_text = tokenizer.decode([pred_token])
        label_text = tokenizer.decode([labels[i][-1]])
        f.write(f"Prediction: {pred_text.strip()} | True: {label_text.strip()}\n")