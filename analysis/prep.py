"""Data preparation utilities for the Random Forest intrusion-detection study.

Project : Using the Random Forest Algorithm to Detect Cyber Threats in Network Traffic Data
Author  : Kresen Naicker (ST10312227)
Module  : RPDA8412 Research Project

The module prepares the two benchmark intrusion-detection corpora used in the study:

  * UNSW-NB15 (primary)  - official train/test partitions (Moustafa & Slay, 2015)
  * CICIDS2017 (secondary robustness check) - stratified sample of the
    MachineLearningCSV flow files (Sharafaldin, Lashkari & Ghorbani, 2018)
"""

import json
import os

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
RESULTS = os.path.join(BASE, "analysis", "results")
FIGS = os.path.join(BASE, "analysis", "figures")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)

RANDOM_STATE = 42

# Categorical (nominal) flow attributes in UNSW-NB15
UNSW_CAT = ["proto", "service", "state"]
UNSW_DROP = ["id", "attack_cat", "label"]

CICIDS_FILES = [
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
]


# --------------------------------------------------------------------------- #
# UNSW-NB15
# --------------------------------------------------------------------------- #
def load_unsw_raw():
    train = pd.read_csv(os.path.join(DATA, "UNSW_NB15_training-set.csv"), low_memory=False)
    test = pd.read_csv(os.path.join(DATA, "UNSW_NB15_testing-set.csv"), low_memory=False)
    train.columns = [c.strip().replace("\ufeff", "") for c in train.columns]
    test.columns = [c.strip().replace("\ufeff", "") for c in test.columns]
    return train, test


def audit_unsw(train, test):
    """Data-quality audit reported in Table 4.1 of the research report."""
    audit = {
        "train_rows": int(len(train)),
        "train_cols": int(train.shape[1]),
        "test_rows": int(len(test)),
        "test_cols": int(test.shape[1]),
        "train_duplicates": int(train.duplicated().sum()),
        "test_duplicates": int(test.duplicated().sum()),
        "train_duplicates_feature_only": int(train.drop(columns=["id"]).duplicated().sum()),
        "test_duplicates_feature_only": int(test.drop(columns=["id"]).duplicated().sum()),
        "train_missing_cells": int(train.isna().sum().sum()),
        "test_missing_cells": int(test.isna().sum().sum()),
        "numeric_features": int(len(train.select_dtypes(include=[np.number]).columns) - 2),
        "categorical_features": int(len(UNSW_CAT)),
        "binary_label_counts_train": {int(k): int(v) for k, v in train["label"].value_counts().items()},
        "binary_label_counts_test": {int(k): int(v) for k, v in test["label"].value_counts().items()},
        "attack_cat_train": {str(k): int(v) for k, v in train["attack_cat"].value_counts().items()},
        "attack_cat_test": {str(k): int(v) for k, v in test["attack_cat"].value_counts().items()},
        "service_placeholder_count_train": int((train["service"] == "-").sum()),
        "service_placeholder_count_test": int((test["service"] == "-").sum()),
        "proto_levels": int(train["proto"].nunique()),
        "service_levels": int(train["service"].nunique()),
        "state_levels": int(train["state"].nunique()),
        "infinite_values_train": int(
            np.isinf(train.select_dtypes(include=[np.number])).sum().sum()
        ),
        "infinite_values_test": int(
            np.isinf(test.select_dtypes(include=[np.number])).sum().sum()
        ),
    }
    const = [c for c in train.columns if c not in ("id", "attack_cat", "label")
             and train[c].nunique(dropna=False) <= 1]
    audit["constant_features"] = const
    audit["near_constant_features"] = [
        c for c in train.columns
        if c not in ("id", "attack_cat", "label")
        and len(const) and c not in const
        and train[c].value_counts(dropna=False).iloc[0] / len(train) > 0.995
    ]
    return audit


def clean_unsw(train, test):
    """Cleaning step: de-duplicate, replace infinities, map '-' placeholders.

    Both partitions are de-duplicated on the full record (features plus both
    label columns).  A copy of the unde-duplicated test partition is retained so
    that the effect of duplicated records can be reported as a sensitivity
    analysis (Section 4.3.7 of the report).
    """
    tr = train.drop(columns=["id"]).copy()
    te = test.drop(columns=["id"]).copy()
    te_raw = te.copy()

    before = len(tr)
    tr = tr.drop_duplicates().reset_index(drop=True)
    dup_removed = before - len(tr)

    before_te = len(te)
    te = te.drop_duplicates().reset_index(drop=True)
    dup_removed_test = before_te - len(te)

    # conflicting labels attached to identical feature vectors (label noise)
    feat_only = [c for c in tr.columns if c not in ("attack_cat", "label")]
    conflict = int(tr.duplicated(subset=feat_only).sum())

    num_cols = [c for c in tr.columns if c not in UNSW_CAT + ["attack_cat", "label"]]
    for df in (tr, te, te_raw):
        df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
        df["service"] = df["service"].replace("-", "none")
        df["state"] = df["state"].replace("-", "none")
    tr[num_cols] = tr[num_cols].fillna(0.0)
    te[num_cols] = te[num_cols].fillna(0.0)
    te_raw[num_cols] = te_raw[num_cols].fillna(0.0)
    return tr, te, te_raw, dup_removed, dup_removed_test, conflict


def encode_unsw(tr, te, te_raw, min_count=50):
    """One-hot encode nominal flow attributes with rare-level grouping."""
    encoded_levels = {}
    for col in UNSW_CAT:
        counts = tr[col].value_counts()
        keep = sorted(counts[counts >= min_count].index)
        encoded_levels[col] = keep
        for df in (tr, te, te_raw):
            df[col] = df[col].where(df[col].isin(keep), other="rare")
    combined = pd.concat([tr[UNSW_CAT], te[UNSW_CAT], te_raw[UNSW_CAT]], axis=0)
    dummies = pd.get_dummies(combined, columns=UNSW_CAT, prefix=UNSW_CAT, dtype=np.int8)
    tr_d = dummies.iloc[: len(tr)].reset_index(drop=True)
    te_d = dummies.iloc[len(tr): len(tr) + len(te)].reset_index(drop=True)
    te_raw_d = dummies.iloc[len(tr) + len(te):].reset_index(drop=True)

    feature_cols = [c for c in tr.columns if c not in UNSW_DROP + UNSW_CAT]
    X_train = pd.concat([tr[feature_cols].reset_index(drop=True), tr_d], axis=1)
    X_test = pd.concat([te[feature_cols].reset_index(drop=True), te_d], axis=1)
    X_test_raw = pd.concat([te_raw[feature_cols].reset_index(drop=True), te_raw_d], axis=1)
    # align
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    X_test_raw = X_test_raw.reindex(columns=X_train.columns, fill_value=0)

    y_train_bin = tr["label"].astype(np.int8).values
    y_test_bin = te["label"].astype(np.int8).values
    y_train_cat = tr["attack_cat"].astype(str).values
    y_test_cat = te["attack_cat"].astype(str).values
    y_test_raw_bin = te_raw["label"].astype(np.int8).values
    return (X_train, X_test, X_test_raw, y_train_bin, y_test_bin, y_test_raw_bin,
            y_train_cat, y_test_cat, encoded_levels)


def build_unsw():
    train, test = load_unsw_raw()
    audit = audit_unsw(train, test)
    tr, te, te_raw, dup_removed, dup_removed_test, conflict = clean_unsw(train, test)
    audit["duplicates_removed_train"] = int(dup_removed)
    audit["duplicates_removed_test"] = int(dup_removed_test)
    audit["train_rows_after_clean"] = int(len(tr))
    audit["test_rows_after_clean"] = int(len(te))
    audit["identical_feature_vectors_train"] = int(conflict)
    audit["binary_label_counts_train_clean"] = {int(k): int(v) for k, v in tr["label"].value_counts().items()}
    audit["binary_label_counts_test_clean"] = {int(k): int(v) for k, v in te["label"].value_counts().items()}
    audit["attack_cat_train_clean"] = {str(k): int(v) for k, v in tr["attack_cat"].value_counts().items()}
    audit["attack_cat_test_clean"] = {str(k): int(v) for k, v in te["attack_cat"].value_counts().items()}
    (X_train, X_test, X_test_raw, ytr, yte, yte_raw, ytr_cat, yte_cat,
     levels) = encode_unsw(tr, te, te_raw)
    audit["encoded_feature_count"] = int(X_train.shape[1])
    audit["categorical_levels_retained"] = {k: len(v) for k, v in levels.items()}

    # persist prepared matrices for the modelling scripts
    X_train.to_pickle(os.path.join(DATA, "unsw_X_train.pkl"))
    X_test.to_pickle(os.path.join(DATA, "unsw_X_test.pkl"))
    X_test_raw.to_pickle(os.path.join(DATA, "unsw_X_test_raw.pkl"))
    np.save(os.path.join(DATA, "unsw_y_test_raw_bin.npy"), yte_raw)
    np.save(os.path.join(DATA, "unsw_y_train_bin.npy"), ytr)
    np.save(os.path.join(DATA, "unsw_y_test_bin.npy"), yte)
    np.save(os.path.join(DATA, "unsw_y_train_cat.npy"), ytr_cat)
    np.save(os.path.join(DATA, "unsw_y_test_cat.npy"), yte_cat)
    with open(os.path.join(RESULTS, "audit_unsw.json"), "w") as fh:
        json.dump(audit, fh, indent=2)
    print(json.dumps(audit, indent=2)[:3000])
    print("X_train", X_train.shape, "X_test", X_test.shape, "X_test_raw", X_test_raw.shape)
    return X_train, X_test, ytr, yte, ytr_cat, yte_cat


# --------------------------------------------------------------------------- #
# CICIDS2017 (secondary dataset)
# --------------------------------------------------------------------------- #
def build_cicids(per_file_cap=30000, chunksize=60000):
    """Read the flow CSVs in chunks and draw a stratified per-file sample.

    Sampling is performed within each source file so that the attack mix of the
    benchmark is preserved while memory use stays inside the available budget.
    """
    frames = []
    file_summary = {}
    for fname in CICIDS_FILES:
        path = os.path.join(DATA, "cicids", fname)
        if not os.path.exists(path):
            continue
        kept = []
        n_total = 0
        for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
            chunk.columns = [c.strip().replace("\ufeff", "") for c in chunk.columns]
            n_total += len(chunk)
            frac = min(1.0, per_file_cap / max(n_total, 1))
            take = max(50, int(len(chunk) * frac))
            if take >= len(chunk):
                kept.append(chunk)
            else:
                kept.append(chunk.sample(n=take, random_state=RANDOM_STATE))
        df = pd.concat(kept, ignore_index=True)
        if len(df) > per_file_cap:
            df = df.sample(n=per_file_cap, random_state=RANDOM_STATE)
        df["source_file"] = fname.replace(".pcap_ISCX.csv", "")
        frames.append(df)
        file_summary[fname] = {"rows_in_file": int(n_total), "rows_sampled": int(len(df))}
        print(f"{fname}: total={n_total} sampled={len(df)}")
    df = pd.concat(frames, ignore_index=True)

    label_col = [c for c in df.columns if c.lower() == "label"][0]
    df["binary"] = (df[label_col].astype(str).str.strip().str.lower() != "benign").astype(np.int8)
    df[label_col] = df[label_col].astype(str).str.strip()

    # CICIDS2017 contains duplicated records and a small number of +/-inf values
    n_before = len(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    obj_cols = [c for c in df.columns if df[c].dtype == object and c not in (label_col, "source_file")]
    for c in obj_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], 0.0)
    df = df.dropna(subset=["binary"])
    df = df.drop_duplicates(subset=[c for c in df.columns if c not in ("source_file",)])
    summary = {
        "rows_sampled_before_dedup": int(n_before),
        "rows_after_dedup": int(len(df)),
        "label_counts": {str(k): int(v) for k, v in df[label_col].value_counts().items()},
        "binary_counts": {int(k): int(v) for k, v in df["binary"].value_counts().items()},
        "files": file_summary,
    }

    drop_cols = [label_col, "source_file"]
    feat = [c for c in df.columns if c not in drop_cols + ["binary"]]
    feature_cols = [c for c in feat if pd.api.types.is_numeric_dtype(df[c])]
    X = df[feature_cols].astype(np.float32)
    X = X.loc[:, X.nunique() > 1]
    # guard against extreme magnitudes for the tree splits
    X = X.replace([np.inf, -np.inf], 0.0)
    y_bin = df["binary"].values
    y_multi = df[label_col].values
    summary["feature_count"] = int(X.shape[1])
    summary["feature_names"] = list(X.columns)
    with open(os.path.join(RESULTS, "audit_cicids.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    X.to_pickle(os.path.join(DATA, "cicids_X.pkl"))
    np.save(os.path.join(DATA, "cicids_y_bin.npy"), y_bin)
    np.save(os.path.join(DATA, "cicids_y_multi.npy"), y_multi)
    print("CICIDS prepared:", X.shape, "positive rate", y_bin.mean().round(4))
    return X, y_bin, y_multi


if __name__ == "__main__":
    import sys

    what = sys.argv[1] if len(sys.argv) > 1 else "unsw"
    if what in ("unsw", "all"):
        build_unsw()
    if what in ("cicids", "all"):
        build_cicids()
