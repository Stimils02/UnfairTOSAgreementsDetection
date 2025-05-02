from unsloth import FastLanguageModel
from transformers import AutoTokenizer

model_name = "unsloth/tinyllama-bnb-4bit"
local_dir = "./local_model/tinyllama-bnb-4bit"

# Make sure you're logged in via CLI or pass token here
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=2048,
    load_in_4bit=True,
    token="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"
)

# Save locally for offline use
model.save_pretrained(local_dir)
tokenizer.save_pretrained(local_dir)