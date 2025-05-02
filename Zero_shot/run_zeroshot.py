import openai
import pandas as pd
from tqdm import tqdm

# ========== CONFIGURATION ==========
# Load dataset
DATASET_PATH = "hf://datasets/LawInformedAI/claudette_tos/data/train-00000-of-00001-a8de7efe0da36666.parquet"
SAMPLE_SIZE = 50  # To avoid high costs — increase if needed
MODEL_NAME = "gpt-4"  # or "gpt-3.5-turbo" for cheaper cost
OPENAI_API_KEY = "your-api-key-here"  # Replace with your actual API key
OUTPUT_CSV = "zero_shot_results.csv"

# ========== SETUP ==========
openai.api_key = OPENAI_API_KEY

# Load dataset
print(f"Loading dataset from {DATASET_PATH}...")
df = pd.read_parquet(DATASET_PATH)
sample_df = df.sample(SAMPLE_SIZE, random_state=42).reset_index(drop=True)

# ========== PROMPT FUNCTION ==========
def create_prompt(clause_text):
    return f"""You are a legal expert. Given the following clause from a Terms of Service document, determine if it is likely **unfair** or **fair** to a user under EU consumer law.

Clause:
\"\"\"{clause_text}\"\"\"

Respond with one word only: "unfair" or "fair".

Answer:"""

# ========== ZERO-SHOT INFERENCE ==========
results = []
print(f"Running zero-shot classification with {MODEL_NAME}...")

for i, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
    clause = row["text"]

    try:
        response = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": create_prompt(clause)}],
            temperature=0,
            max_tokens=1
        )
        label = response['choices'][0]['message']['content'].strip().lower()
    except Exception as e:
        label = "error"
        print(f"[!] Error on row {i}: {e}")

    results.append({
        "original_text": clause,
        "predicted_label": label
    })

# ========== SAVE RESULTS ==========
output_df = pd.DataFrame(results)
output_df.to_csv(OUTPUT_CSV, index=False)
print(f"Results saved to: {OUTPUT_CSV}")