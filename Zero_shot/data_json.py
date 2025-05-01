import os
import json
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("coastalcph/lex_glue", "ecthr_a", split="test")
label_names = dataset.features["labels"].feature.names

# Prepare the data
formatted_data = []
for example in dataset:
    formatted_data.append({
        "input": " ".join(example["text"]),
        "label": [label_names[label] for label in example["labels"]],
        "options": label_names
    })

# Save to JSONL
os.makedirs("lex-glue/ecthr_a", exist_ok=True)
with open("lex-glue/ecthr_a/test.jsonl", "w") as f:
    for entry in formatted_data:
        f.write(json.dumps(entry) + "\n")