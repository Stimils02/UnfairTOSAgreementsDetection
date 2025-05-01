# ------------------------------
# Unified Zero-Shot Runner (LegalNLP)
# ------------------------------
import json
import openai
from sklearn.metrics import f1_score
from tqdm import tqdm

# ------------------------------
# Load LexGLUE data from local JSONL
# ------------------------------
def load_local_lexglue_jsonl(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

# ------------------------------
# Construct zero-shot prompt
# ------------------------------
def make_zeroshot_prompt(text, labels):
    label_list = "\n".join([f"{i+1}. {label}" for i, label in enumerate(labels)])
    return f"""
You are a legal expert. Categorize the text below:

{text}

Choose the most suitable category from:
{label_list}
"""

# ------------------------------
# Query OpenAI GPT-3.5
# ------------------------------

import os
openai.api_key = os.getenv("OPENAI_API_KEY")

def query_gpt(prompt, model="gpt-3.5-turbo"):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message["content"].strip()

# ------------------------------
# Evaluate using F1 Score
# ------------------------------
def evaluate(true_labels, predicted_labels):
    return f1_score(true_labels, predicted_labels, average="micro")

# ------------------------------
# Main Zero-Shot Evaluation Loop
# ------------------------------
def run_zero_shot(jsonl_path, num_examples=50):
    dataset = load_local_lexglue_jsonl(jsonl_path)
    label_set = sorted(list({label for ex in dataset for label in ex["options"]}))

    true_labels = []
    pred_labels = []

    for ex in tqdm(dataset[:num_examples]):
        prompt = make_zeroshot_prompt(ex["input"], label_set)
        response = query_gpt(prompt)
        pred_labels.append(response)
        true_labels.append(ex["label"])

    score = evaluate(true_labels, pred_labels)
    print("Zero-shot GPT micro-F1:", score)

if __name__ == "__main__":
    run_zero_shot("lex-glue/eurlex/test.jsonl")