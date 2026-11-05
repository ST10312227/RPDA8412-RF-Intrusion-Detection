"""CICIDS2017 replication experiment (secondary dataset).

Repeats the core binary-detection, feature-selection and timing measurements on
a stratified sample of the CICIDS2017 flow corpus in order to test whether the
UNSW-NB15 findings are dataset-specific.
"""

import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, classification_report,
                             cohen_kappa_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
RESULTS = os.path.join(BASE, "analysis", "results")
RS = 42


def metrics(y, p, prob=None):
    cm = confusion_matrix(y, p)
    tn, fp, fn, tp = cm.ravel()
    m = {"accuracy": float(accuracy_score(y, p)),
         "precision": float(precision_score(y, p, zero_division=0)),
         "recall": float(recall_score(y, p, zero_division=0)),
         "f1": float(f1_score(y, p, zero_division=0)),
         "fpr": float(fp / (fp + tn)),
         "balanced_accuracy": float(balanced_accuracy_score(y, p)),
         "mcc": float(matthews_corrcoef(y, p)),
         "kappa": float(cohen_kappa_score(y, p)),
         "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    if prob is not None:
        m["roc_auc"] = float(roc_auc_score(y, prob))
        m["pr_auc"] = float(average_precision_score(y, prob))
    return m


def main():
    X = pd.read_pickle(os.path.join(DATA, "cicids_X.pkl"))
    y = np.load(os.path.join(DATA, "cicids_y_bin.npy"))
    y_multi = np.load(os.path.join(DATA, "cicids_y_multi.npy"), allow_pickle=True)
    print("CICIDS", X.shape, "attack rate", round(float(y.mean()), 4))

    Xtr, Xte, ytr, yte, ytr_m, yte_m = train_test_split(
        X, y, y_multi, test_size=0.3, random_state=RS, stratify=y)

    out = {"n_total": int(len(X)), "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
           "attack_rate": float(y.mean()), "n_features": int(X.shape[1]),
           "label_counts": {str(k): int(v) for k, v in
                            pd.Series(y_multi).value_counts().items()}}

    t = time.time()
    rf = RandomForestClassifier(n_estimators=200, random_state=RS, n_jobs=2,
                                oob_score=True, max_features="sqrt")
    rf.fit(Xtr, ytr)
    out["train_time_s"] = time.time() - t
    out["oob_score"] = float(rf.oob_score_)
    prob = rf.predict_proba(Xte)[:, 1]
    out["baseline"] = metrics(yte, rf.predict(Xte), prob)
    print("CICIDS baseline:", {k: round(v, 4) for k, v in out["baseline"].items()})

    # feature importance and reduced-feature model
    imp = rf.feature_importances_
    order = np.argsort(imp)[::-1]
    names = list(X.columns)
    out["feature_importance"] = [{"feature": names[i], "importance": float(imp[i])}
                                 for i in order[:25]]
    feats = {}
    for k in [5, 10, 15, 20, 30]:
        cols = [names[i] for i in order[:k]]
        m = RandomForestClassifier(n_estimators=200, random_state=RS, n_jobs=2)
        t = time.time()
        m.fit(Xtr[cols], ytr)
        tt = time.time() - t
        p = m.predict_proba(Xte[cols])[:, 1]
        met = metrics(yte, m.predict(Xte[cols]), p)
        met["train_time_s"] = tt
        met["n_features"] = k
        feats[k] = met
        print(f"  CICIDS FS k={k}: f1={met['f1']:.4f} fpr={met['fpr']:.4f}")
    out["feature_selection"] = feats

    # timing on the full-feature model
    batch_times = []
    for _ in range(3):
        s = time.perf_counter()
        rf.predict(Xte)
        batch_times.append(time.perf_counter() - s)
    batch = min(batch_times)
    s = time.perf_counter()
    for i in range(500):
        rf.predict(Xte.iloc[[i]])
    single = (time.perf_counter() - s) / 500
    out["timing"] = {"batch_total_s": batch,
                     "per_flow_us": batch / len(Xte) * 1e6,
                     "throughput_flows_per_s": len(Xte) / batch,
                     "single_record_latency_us": single * 1e6}

    # multiclass (attack-name) performance for the dominant categories
    keep_classes = [c for c in pd.Series(ytr_m).value_counts().index
                    if (pd.Series(yte_m) == c).sum() >= 100]
    mask_tr = np.isin(ytr_m, keep_classes)
    mask_te = np.isin(yte_m, keep_classes)
    mc = RandomForestClassifier(n_estimators=200, random_state=RS, n_jobs=2,
                                class_weight="balanced_subsample")
    mc.fit(Xtr[mask_tr], ytr_m[mask_tr])
    pred_mc = mc.predict(Xte[mask_te])
    out["multi_accuracy"] = float(accuracy_score(yte_m[mask_te], pred_mc))
    out["multi_macro_f1"] = float(f1_score(yte_m[mask_te], pred_mc, average="macro"))
    out["multi_weighted_f1"] = float(f1_score(yte_m[mask_te], pred_mc, average="weighted"))
    out["multi_per_class"] = classification_report(yte_m[mask_te], pred_mc,
                                                   output_dict=True, zero_division=0)
    out["multi_classes"] = keep_classes

    with open(os.path.join(RESULTS, "cicids_results.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(json.dumps({k: round(v, 4) for k, v in out["baseline"].items()}, indent=2))
    print("saved")


if __name__ == "__main__":
    main()
