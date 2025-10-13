# Smart Product Pricing Challenge - 1-Page Methodology

## Introduction
Briefly describe the goal: predict product prices from `catalog_content` and product images using a compliant, <8B parameter multi‑modal pipeline.

## Data Preprocessing
- Describe loading `dataset/train.csv`, `dataset/test.csv`.
- Text cleaning: lowercasing, HTML removal, preserve numeric quantities (IPQ).
- Outlier handling for `price` (capping or removal), log transform for skew.
- Image download strategy using `src/utils.py` with retries and caching.

## Feature Engineering
- Text features: TF‑IDF baseline, later SentenceTransformer (Apache 2.0) embeddings.
- Image features: CLIP/EfficientNet embeddings (MIT/Apache 2.0).
- Structured: IPQ extraction, text length, etc.

## Model Architecture
- Baseline: TF‑IDF + Ridge/LightGBM regression on log(price).
- Advanced: Concatenate text + image embeddings; train LightGBM/MLP; optional ensembling.
- Ensure parameter budget and license compliance.

## Training and Evaluation
- Split: 80/20 train/validation.
- Target: log(price); exponentiate at inference.
- Metric: SMAPE (report % and fraction). Include validation results.
- Hyperparameter tuning (grid/Optuna).

## Inference Pipeline
- Batch encode text/images; cache embeddings.
- Ensure positive predictions; clip to realistic range.
- Output `test_out.csv` with `sample_id,price`.

## Hardware and Environment
- List hardware (CPU/GPU), Python version, and key packages.

## Ethics & Compliance
- Use only provided data; no external price lookup.
- Licenses: MIT/Apache 2.0 models only.

## Appendix (Optional)
- Failure cases, ablations, and future work.
