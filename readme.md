ML Challenge 2025 - Smart Product Pricing

Quickstart

1. Place data files in one of the following (auto-detected):
	- student_resource/dataset/train.csv, test.csv
	- dataset/train.csv, dataset/test.csv
	- Or set env var DATA_DIR to a folder containing train.csv and test.csv

2. (Optional) Create venv and install deps:

	python3 -m venv .venv
	source .venv/bin/activate
	pip install -r requirements.txt

3. Run baseline training + inference (TF‑IDF + Ridge):

	python src/train_baseline.py

4. Outputs:
	- test_out.csv at repo root with columns: sample_id, price

Notes
- Utilities for image download and SMAPE are in src/utils.py
- Do not use external price lookups; use only provided datasets.


![alt text](https://github.com/gaxxrav/amazon/blob/master/diagram.png)
