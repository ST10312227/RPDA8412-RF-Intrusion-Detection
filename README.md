# Using the Random Forest Algorithm to Detect Cyber Threats in Network Traffic Data

**Author:** Kresen Naicker (`ST10312227`)  
**Module:** RPDA8412/w — Postgraduate Diploma in Data Analytics, The Independent Institute of Education  
**Supervisor:** Dr Brian Maodza  
**Ethics Clearance:** `CSREC/ST10312227/104`

## Overview
Companion code, pre-executed Jupyter Notebook, metric outputs (`JSON`), and publication figures for the RPDA8412 summative research report evaluating Random Forest for network intrusion detection on **UNSW-NB15** (primary corpus, 257,673 distributed flows; 163,685 unique flows after de-duplication) and **CICIDS2017** (stratified replication, 146,370 unique flows).

## Repository Contents
- `Naicker_Kresen_ST10312227_RF_Intrusion_Detection.ipynb` — Consolidated Jupyter Notebook with executed outputs and embedded figures (renders directly on GitHub).
- `analysis/prep.py` — Data-quality audit, duplicate removal, placeholder harmonisation, one-hot encoding, and partition preparation.
- `analysis/exp_unsw.py` — Controlled UNSW-NB15 experiments (baseline, imbalance, feature selection, tuning, multiclass, 5-fold CV, single tree, latency).
- `analysis/exp_smote.py` — Matched 60,000-flow subsample SMOTE experiment.
- `analysis/exp_reference.py` — Non-deduplicated reference pipeline quantifying duplicate-induced metric inflation.
- `analysis/exp_cicids.py` — Full pipeline replication on CICIDS2017.
- `analysis/figures.py` — Reproducible generation of Figures 3.1 and 4.1–4.13.
- `analysis/results/*.json` — Verified numerical outputs backing every table in the report.
- `data/download_datasets.py` — One-command fetcher for the public UNSW-NB15 and CICIDS2017 CSV files.

## Reproducing From Scratch
```bash
pip install -r requirements.txt
python3 data/download_datasets.py
python3 analysis/prep.py all
python3 analysis/exp_unsw.py
python3 analysis/exp_smote.py
python3 analysis/exp_reference.py
python3 analysis/exp_cicids.py
python3 analysis/figures.py
```
