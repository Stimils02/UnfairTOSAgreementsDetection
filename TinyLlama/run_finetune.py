import torch
from unsloth import FastLanguageModel
from datasets import load_from_disk
from transformers import TrainingArguments
from trl import SFTTrainer
import numpy as np

# Load datasets
train_dataset = load_from_disk("data/train")
val_dataset = load_from_disk("data/val")

# Load model + tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/tinyllama-bnb-4bit",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True
)

# Apply LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    use_gradient_checkpointing=True,
)

# Format prompt
def format_example(example):
    prompt = f"<s>[CLAUSE]: {example['text']} \n[Is this anomalous?]:"
    label = " Yes" if example['label'] == 1 else " No"
    return {"text": prompt + label}

train_dataset = train_dataset.map(format_example)
val_dataset = val_dataset.map(format_example)

# Tokenize
def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=256)

train_dataset = train_dataset.map(tokenize)
val_dataset = val_dataset.map(tokenize)

# Training
training_args = TrainingArguments(
    output_dir="./tinyllama-finetuned",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
    fp16=torch.cuda.is_available()
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    args=training_args,
)

trainer.train()

# Evaluation
predictions = trainer.predict(val_dataset)
pred_labels = np.argmax(predictions.predictions, axis=-1)
true_labels = predictions.label_ids

for i in range(5):
    print(f"Prediction: {pred_labels[i]} | True label: {true_labels[i]} | Text: {val_dataset[i]['text']}")