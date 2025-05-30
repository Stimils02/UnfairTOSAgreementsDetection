# Text to Trust: Evaluating Language Model Trade-offs for Unfair Terms of Service Detection

A comprehensive study and implementation of clause-level unfairness detection in Terms of Service (ToS) documents, comparing full fine-tuning, parameter-efficient LoRA tuning, and zero-shot prompting across multiple models, with deployment on real-world web-scale corpora.

---

## Table of Contents

* [Introduction](#introduction)
* [Experiments](#experiments)
  * [Datasets](#datasets)
  * [Baseline Models](#baseline-models)
  * [Approaches](#approaches)
* [Results](#results)
* [Deployment](#deployment)
* [Contributing](#contributing)
* [License](#license)
* [Acknowledgements](#acknowledgements)

---

## Introduction

Terms of Service agreements often contain clauses that are difficult to interpret and potentially unfair to users. Manual review is infeasible at scale, motivating automated, accurate, and efficient detection methods. This project evaluates three modeling paradigms for ToS clause fairness detection:

1. **Full Fine-Tuning** of transformers (BERT, DistilBERT)
2. **Parameter-Efficient LoRA Tuning** with 4-bit quantization (TinyLlama, LLaMA, SaulLM)
3. **Zero-Shot Prompting** using API-accessible LLMs (GPT-4o, O3-mini, etc.)

Our final best-performing classifier is deployed on a multilingual scraped ToS corpus to demonstrate real-world applicability.

## Experiments

### Datasets

* **CLAUDETTE-ToS**: 9,414 English clauses balanced to 50/50 fair vs. unfair.
* **Multilingual Scraped ToS**: \~60GB of clauses scraped from thousands of websites, filtered to English via metadata.

### Baseline Models

* **Fully Fine-Tuned**: BERT (110M params), DistilBERT (66M params).
* **LoRA + Quantization**: TinyLlama-1.1B, LLaMA-3B/7B, SaulLM-7B (4-bit).
* **Zero-Shot Prompting**: GPT-4o, GPT-4o-mini, O1-mini, O3-mini, O4-mini via API.

### Approaches

* **Full Fine-Tuning**: All model weights updated using cross-entropy loss.
* **Parameter-Efficient Fine-Tuning**: LoRA adapters injected into attention layers with 4-bit quantization (PEFT + bitsandbytes).
* **Zero-Shot Prompting**: Standardized system prompt across models, batched inference with post-processing to extract binary labels.

---

## Results

| Model               | Accuracy | Precision | Recall | F1    |
| ------------------- | -------- | --------- | ------ | ----- |
| BERT (FT)           | 88.9%    | 89.2%     | 89.2%  | 89.2% |
| DistilBERT (FT)     | 89.0%    | 89.9%     | 89.3%  | 89.6% |
| TinyLlama + LoRA    | 73.0%    | 89.1%     | 52.5%  | 66.1% |
| SaulLM + LoRA       | 82.3%    | 73.6%     | 97.5%  | 83.9% |
| O3-mini (Zero-shot) | 85.7%    | 42.7%     | 91.3%  | 58.2% |

Full tables and figures are available in `results/figures/` and the final report (PDF).

---

## Deployment

Batch inference on 937 English clauses from the scraped corpus identified 152 high-confidence unfair clauses (≥80% true positive rate after heuristic filtering). See `results/deploy_predictions.xlsx` for details.

---

## Contributing

Contributions, issues, and feature requests are welcome! Please fork the repository and open a pull request.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

* **Project Leads**: Noshitha Juttu, Sahithi Singireddy, Sravani Gona, Sujal Timilsina
* **Advisors**: UMass Amherst COMPSCI 696DS Instructors
* **Datasets**: CLAUDETTE-ToS (Hugging Face), Multilingual Scraper Corpus
