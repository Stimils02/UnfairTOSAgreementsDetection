import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from collections import Counter

# ======= Load the CSV =======
csv_path = "/home/njuttu_umass_edu/685/ZeroShotAnomolyDetection-1/TinyLlama/data/filtered_uk_terms_all_months.csv"
df = pd.read_csv(csv_path)

# ======= Define Prompt Formatter =======
def format_prompt(text):
    return f"<s>[CLAUSE]: {text} \n[Is this anomalous?]:"

# ======= Define Dataset Class =======
class ClauseDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=256):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        prompt = format_prompt(self.texts[idx])
        inputs = self.tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length)
        return {key: val.squeeze(0) for key, val in inputs.items()}

# ======= Load Tokenizer and Base Model =======
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map="auto"
)

# ======= Load LoRA Adapter =======
adapter_path = "/home/njuttu_umass_edu/685/ZeroShotAnomolyDetection/TinyLlama/lora-tinyllama/checkpoint-30"
model = PeftModel.from_pretrained(model, adapter_path)

# ======= Prepare DataLoader =======
dataset = ClauseDataset(df['terms_content'].tolist(), tokenizer)

dataloader = DataLoader(dataset, batch_size=1)

# ======= Run Inference =======
model.eval()
predictions = []

for batch in dataloader:
    input_ids = batch["input_ids"].unsqueeze(0).to(model.device)
    attention_mask = batch["attention_mask"].unsqueeze(0).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=5,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Get only generated part (after prompt)
    generated_tokens = outputs[0][input_ids.shape[1]:]
    decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip().lower()

    if "yes" in decoded:
        predictions.append("Unfair")
    else:
        predictions.append("Fair")

# ======= Save Results =======
df["Fairness Prediction"] = predictions
df.to_csv("predicted_fairness.csv", index=False)
print("Predictions saved to predicted_fairness.csv")

# Summary Report
summary = Counter(predictions)
print("\n Classification Summary ")
print(f"Total clauses analyzed: {len(predictions)}")
print(f"Fair: {summary['Fair']}")
print(f"Unfair: {summary['Unfair']}")