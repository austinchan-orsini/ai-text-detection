# ai-text-detection
Repo for CSCI3345 final project: Do AI-text detectors generalize to unseen generators?

## Setup
pip install -r requirements.txt

## Data
git clone --depth 1 https://github.com/mbzuai-nlp/M4.git

## Reproduce
python src/prepare_data.py --m4_dir M4/data --out data/m4_subset.parquet
python src/baseline.py --data data/m4_subset.parquet --out results/

All results are on the validation split; the test split is untouched.
