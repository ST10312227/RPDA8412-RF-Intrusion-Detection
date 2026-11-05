"""Reference (literature-standard) pipeline: train on the raw, non-deduplicated
UNSW-NB15 training partition and evaluate on the raw test partition.

Purpose: quantify how the data-quality treatment used in this study (duplicate
removal) changes the headline performance figures that are commonly reported in
published UNSW-NB15 studies.  This supports the data-quality argument in
Sections 4.3.7 and 4.5 of the report.
"""

import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
RESULTS = os.path.join(BASE, "analysis", "results")
RS = 42


def encode_raw(train, test):
    tr = train.drop(columns=["id"]).copy()
    te = test.drop(columns=["id"]).copy()
    for df in (tr, te):
        df["service"] = df["service"].replace("-", "none")
        df["state"] = df["state"].replace("-", "none")
    cat = ["proto", "service", "state"]
    counts = tr["proto"].value_counts()
    keep = sorted(counts[counts >= 50].index)
    for df in (tr, te):
        df["proto"] = df["proto"].where(df["proto"].isin(keep), other="rare")
    combined = pd.concat([tr[cat], te[cat]], axis=0)
    dum = pd.get_dummies(combined, columns=cat, prefix=cat, dtype=np.int8)
    tr_d = dum.iloc[: len(tr)].reset_index(drop=True)
    te_d = dum.iloc[len(tr):].reset_index(drop=True)
    feats = [c for c in tr.columns if c not in cat + ["attack_cat", "label"]]
    Xtr = pd.concat([tr[feats].reset_index(drop=True), tr_d], axis=1)
    Xte = pd.concat([te[feats].reset_index(drop=True), te_d], axis=1)
    Xte = Xte.reindex(columns=Xtr.columns, fill_value=0)
    return Xtr, Xte, tr["label"].values, te["label"].values


def metrics(y, p):
    cm = confusion_matrix(y, p)
    tn, fp, fn, tp = cm.ravel()
    return {"accuracy": float(accuracy_score(y, p)),
            "precision": float(precision_score(y, p)),
            "recall": float(recall_score(y, p)),
            "f1": float(f1_score(y, p)),
            "fpr": float(fp / (fp + tn)),
            "mcc": float(matthews_corrcoef(y, p)),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}


def main():
    train = pd.read_csv(os.path.join(DATA, "UNSW_NB15_training-set.csv"), low_memory=False)
    test = pd.read_csv(os.path.join(DATA, "UNSW_NB15_testing-set.csv"), low_memory=False)
    train.columns = [c.strip().replace("\ufeff", "") for c in train.columns]
    test.columns = [c.strip().replace("\ufeff", "") for c in test.columns]
    Xtr, Xte, ytr, yte = encode_raw(train, test)
    print("raw pipeline shapes", Xtr.shape, Xte.shape)

    out = {"n_train_raw": int(len(Xtr)), "n_test_raw": int(len(Xte))}
    t = time.time()
    rf = RandomForestClassifier(n_estimators=200, random_state=RS, n_jobs=2,
                                oob_score=True, max_features="sqrt")
    rf.fit(Xtr, ytr)
    out["train_time_s"] = time.time() - t
    out["oob_score"] = float(rf.oob_score_)
    prob = rf.predict_proba(Xte)[:, 1]
    out["raw_pipeline_raw_test"] = metrics(yte, rf.predict(Xte))
    out["raw_pipeline_raw_test"]["roc_auc"] = float(roc_auc_score(yte, prob))

    # evaluate the same model on the de-duplicated test partition
    ded = pd.read_pickle(os.path.join(DATA, "unsw_X_test.pkl"))
    y_ded = np.load(os.path.join(DATA, "unsw_y_test_bin.npy"))
    ded = ded.reindex(columns=Xtr.columns, fill_value=0)
    p2 = rf.predict(ded)
    pr2 = rf.predict_proba(ded)[:, 1]
    out["raw_pipeline_dedup_test"] = metrics(y_ded, p2)
    out["raw_pipeline_dedup_test"]["roc_auc"] = float(roc_auc_score(y_ded, pr2))
    print(json.dumps(out, indent=2))
    with open(os.path.join(RESULTS, "reference_pipeline.json"), "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
