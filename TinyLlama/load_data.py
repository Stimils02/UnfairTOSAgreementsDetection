from datasets import load_dataset, Dataset
import pandas as pd

# Load and prepare the dataset
ds = load_dataset("LawInformedAI/claudette_tos")
df = pd.DataFrame(ds["train"])

# Balance the dataset
anomalous = df[df['label'] == 1].sample(n=50, random_state=42)
normal = df[df['label'] == 0].sample(n=50, random_state=42)
balanced_df = pd.concat([anomalous, normal]).sample(frac=1, random_state=42).reset_index(drop=True)

# Split
train_df, val_df = balanced_df[:80], balanced_df[80:]
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Save datasets to disk (for run_finetune.py)
train_dataset.save_to_disk("data/train")
val_dataset.save_to_disk("data/val")