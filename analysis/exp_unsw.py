"""Random Forest experiments on UNSW-NB15 - stage runner with checkpoints.

Each stage writes its own JSON file so that a failure in one stage never
destroys completed work.  Run as:  python3 exp_unsw.py <stage>
Stages: base | imbalance | features | tuning | multiclass | timing | cv | all
"""

import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, classification_report,
                             cohen_kappa_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     cross_validate)
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
RESULTS = os.path.join(BASE, "analysis", "results")
MODELS = os.path.join(BASE, "analysis", "models")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)
RS = 42
NTREES = 200


def load(as_float32=True):
    Xtr = pd.read_pickle(os.path.join(DATA, "unsw_X_train.pkl"))
    Xte = pd.read_pickle(os.path.join(DATA, "unsw_X_test.pkl"))
    Xraw = pd.read_pickle(os.path.join(DATA, "unsw_X_test_raw.pkl"))
    ytr = np.load(os.path.join(DATA, "unsw_y_train_bin.npy"))
    yte = np.load(os.path.join(DATA, "unsw_y_test_bin.npy"))
    yraw = np.load(os.path.join(DATA, "unsw_y_test_raw_bin.npy"))
    ytr_cat = np.load(os.path.join(DATA, "unsw_y_train_cat.npy"), allow_pickle=True)
    yte_cat = np.load(os.path.join(DATA, "unsw_y_test_cat.npy"), allow_pickle=True)
    if as_float32:
        Xtr = Xtr.astype(np.float32)
        Xte = Xte.astype(np.float32)
        Xraw = Xraw.astype(np.float32)
    return Xtr, Xte, Xraw, ytr, yte, yraw, ytr_cat, yte_cat


def save(name, obj):
    with open(os.path.join(RESULTS, name), "w") as fh:
        json.dump(obj, fh, indent=2, default=str)
    print("saved", name)


def bmetrics(y, pred, prob):
    cm = confusion_matrix(y, pred)
    tn, fp, fn, tp = cm.ravel()
    return {"accuracy": float(accuracy_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, prob)),
            "pr_auc": float(average_precision_score(y, prob)),
            "fpr": float(fp / (fp + tn)),
            "fnr": float(fn / (fn + tp)),
            "specificity": float(tn / (tn + fp)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "mcc": float(matthews_corrcoef(y, pred)),
            "kappa": float(cohen_kappa_score(y, pred)),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "n": int(len(y))}


# --------------------------------------------------------------------------- #
def stage_base():
    Xtr, Xte, Xraw, ytr, yte, yraw, *_ = load()
    out = {"n_train": int(len(Xtr)), "n_test": int(len(Xte)), "n_test_raw": int(len(Xraw)),
           "n_features": int(Xtr.shape[1]), "features": list(Xtr.columns)}

    t = time.time()
    rf = RandomForestClassifier(n_estimators=NTREES, random_state=RS, n_jobs=2,
                                oob_score=True, max_features="sqrt")
    rf.fit(Xtr.values, ytr)
    out["train_time_s"] = time.time() - t
    out["oob_score"] = float(rf.oob_score_)

    t = time.perf_counter()
    prob = rf.predict_proba(Xte.values)[:, 1]
    pred = rf.predict(Xte.values)
    out["predict_time_full_test_s"] = time.perf_counter() - t
    out["test"] = bmetrics(yte, pred, prob)

    prob_raw = rf.predict_proba(Xraw.values)[:, 1]
    out["test_raw"] = bmetrics(yraw, rf.predict(Xraw.values), prob_raw)

    prob_tr = rf.predict_proba(Xtr.values)[:, 1]
    out["train"] = bmetrics(ytr, rf.predict(Xtr.values), prob_tr)

    out["n_nodes"] = int(sum(e.tree_.node_count for e in rf.estimators_))
    out["mean_tree_depth"] = float(np.mean([e.tree_.max_depth for e in rf.estimators_]))

    imp = rf.feature_importances_
    order = np.argsort(imp)[::-1]
    out["feature_importance"] = [{"feature": Xtr.columns[i], "importance": float(imp[i])}
                                 for i in order]
    out["cumulative_importance"] = {f"top{k}": float(np.cumsum(imp[order])[k - 1])
                                    for k in (5, 10, 15, 20, 30, 50)}

    # threshold sweep + curves
    fpr, tpr, thr = roc_curve(yte, prob)
    pc, rc, _ = precision_recall_curve(yte, prob)
    out["roc"] = {"fpr": fpr[::15].tolist() + [1.0], "tpr": tpr[::15].tolist() + [1.0]}
    out["pr"] = {"recall": rc[::15].tolist(), "precision": pc[::15].tolist()}
    sweep = []
    for th in np.arange(0.05, 0.991, 0.05):
        p = (prob >= th).astype(int)
        cm = confusion_matrix(yte, p)
        tn, fp, fn, tp = cm.ravel()
        sweep.append({"threshold": round(float(th), 2),
                      "precision": float(precision_score(yte, p, zero_division=0)),
                      "recall": float(recall_score(yte, p, zero_division=0)),
                      "f1": float(f1_score(yte, p, zero_division=0)),
                      "fpr": float(fp / (fp + tn)),
                      "alerts_per_10k": float((tp + fp) / len(yte) * 10000),
                      "false_alerts_per_10k_benign": float(fp / max(tn + fp, 1) * 10000)})
    out["threshold_sweep"] = sweep
    ops = {}
    for target in (0.90, 0.95, 0.98, 0.99):
        idx = np.where(tpr >= target)[0]
        if len(idx):
            i = idx[0]
            ops[f"recall_{target}"] = {"threshold": float(thr[i]), "fpr": float(fpr[i])}
    out["operating_points"] = ops
    # FPR needed to reach a fixed alert budget
    for budget in (0.01, 0.05, 0.10):
        idx = np.where(fpr <= budget)[0]
        if len(idx):
            i = idx[-1]
            out[f"at_fpr_{budget}"] = {"threshold": float(thr[i]), "recall": float(tpr[i])}

    save("unsw_base.json", out)
    dump(rf, os.path.join(MODELS, "rf_baseline.joblib"), compress=0)
    print({k: round(v, 4) for k, v in out["test"].items() if isinstance(v, float)})
    return out


# --------------------------------------------------------------------------- #
def stage_imbalance():
    Xtr, Xte, *_rest = load()
    ytr = np.load(os.path.join(DATA, "unsw_y_train_bin.npy"))
    yte = np.load(os.path.join(DATA, "unsw_y_test_bin.npy"))
    out = {}
    t = time.time()
    rf = RandomForestClassifier(n_estimators=NTREES, random_state=RS, n_jobs=2,
                                class_weight="balanced", max_features="sqrt")
    rf.fit(Xtr.values, ytr)
    out["balanced"] = bmetrics(yte, rf.predict(Xte.values), rf.predict_proba(Xte.values)[:, 1])
    out["balanced_train_time_s"] = time.time() - t
    print("balanced", {k: round(v, 4) for k, v in out["balanced"].items() if isinstance(v, float)})
    save("unsw_imbalance.json", out)

    # SMOTE - applied to a 60k stratified subsample to stay inside the RAM budget
    from imblearn.over_sampling import SMOTE
    rng = np.random.RandomState(RS)
    idx = rng.choice(len(Xtr), size=60000, replace=False)
    Xs, ys = Xtr.values[idx], ytr[idx]
    t = time.time()
    Xres, yres = SMOTE(random_state=RS, sampling_strategy=0.9,
                       k_neighbors=5).fit_resample(Xs, ys)
    out["smote_train_size"] = int(len(Xres))
    out["smote_train_class_counts"] = {int(k): int(v) for k, v in zip(*np.unique(yres, return_counts=True))}
    rf = RandomForestClassifier(n_estimators=NTREES, random_state=RS, n_jobs=2, max_features="sqrt")
    rf.fit(Xres, yres)
    out["smote_train_time_s"] = time.time() - t
    out["smote_subset_baseline"] = None
    rf0 = RandomForestClassifier(n_estimators=NTREES, random_state=RS, n_jobs=2, max_features="sqrt")
    rf0.fit(Xs, ys)
    out["smote_subset_baseline"] = bmetrics(yte, rf0.predict(Xte.values),
                                            rf0.predict_proba(Xte.values)[:, 1])
    out["smote"] = bmetrics(yte, rf.predict(Xte.values), rf.predict_proba(Xte.values)[:, 1])
    print("smote", {k: round(v, 4) for k, v in out["smote"].items() if isinstance(v, float)})
    save("unsw_imbalance.json", out)


# --------------------------------------------------------------------------- #
def stage_features():
    Xtr, Xte, Xraw, ytr, yte, yraw, ytr_cat, yte_cat = load()
    base = json.load(open(os.path.join(RESULTS, "unsw_base.json")))
    order = [d["feature"] for d in base["feature_importance"]]
    names = list(Xtr.columns)
    out = {"ranking": order}

    fs = {}
    for k in (5, 10, 15, 20, 30, 50):
        cols = order[:k]
        t = time.time()
        m = RandomForestClassifier(n_estimators=NTREES, random_state=RS, n_jobs=2, max_features="sqrt")
        m.fit(Xtr[cols].values, ytr)
        tt = time.time() - t
        s = time.perf_counter()
        p = m.predict_proba(Xte[cols].values)[:, 1]
        pred = m.predict(Xte[cols].values)
        pt = time.perf_counter() - s
        met = bmetrics(yte, pred, p)
        met.update({"n_features": k, "train_time_s": tt, "predict_time_s": pt,
                    "features": cols})
        fs[k] = met
        print(f"FS k={k} f1={met['f1']:.4f} fpr={met['fpr']:.4f} t={tt:.1f}s")
        save("unsw_features.json", {"ranking": order, "importance_selection": fs})
    out["importance_selection"] = fs

    # mutual-information ranking at matched counts
    from sklearn.feature_selection import mutual_info_classif
    rng = np.random.RandomState(RS)
    idx = rng.choice(len(Xtr), size=30000, replace=False)
    mi = mutual_info_classif(Xtr.values[idx], ytr[idx], random_state=RS, n_neighbors=3)
    mi_order = [names[i] for i in np.argsort(mi)[::-1]]
    out["mi_ranking"] = mi_order
    mi_fs = {}
    for k in (10, 20, 30, 50):
        cols = mi_order[:k]
        m = RandomForestClassifier(n_estimators=NTREES, random_state=RS, n_jobs=2, max_features="sqrt")
        m.fit(Xtr[cols].values, ytr)
        met = bmetrics(yte, m.predict(Xte[cols].values), m.predict_proba(Xte[cols].values)[:, 1])
        met["n_features"] = k
        mi_fs[k] = met
        print(f"MI k={k} f1={met['f1']:.4f} fpr={met['fpr']:.4f}")
        save("unsw_features.json", {**out, "mi_selection": mi_fs})
    out["mi_selection"] = mi_fs

    # correlation pruning
    corr = Xtr.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop = [c for c in upper.columns if (upper[c] > 0.95).any()]
    keep = [c for c in names if c not in drop]
    t = time.time()
    m = RandomForestClassifier(n_estimators=NTREES, random_state=RS, n_jobs=2, max_features="sqrt")
    m.fit(Xtr[keep].values, ytr)
    tt = time.time() - t
    met = bmetrics(yte, m.predict(Xte[keep].values), m.predict_proba(Xte[keep].values)[:, 1])
    met.update({"n_features": len(keep), "dropped": drop, "train_time_s": tt})
    out["correlation_pruning"] = met
    print("corr", len(keep), round(met["f1"], 4))
    save("unsw_features.json", out)

    # hybrid: correlation pruning then importance top-20
    imp2 = m.feature_importances_
    top20 = [keep[i] for i in np.argsort(imp2)[::-1][:20]]
    m2 = RandomForestClassifier(n_estimators=NTREES, random_state=RS, n_jobs=2, max_features="sqrt")
    m2.fit(Xtr[top20].values, ytr)
    met = bmetrics(yte, m2.predict(Xte[top20].values), m2.predict_proba(Xte[top20].values)[:, 1])
    met.update({"n_features": 20, "features": top20})
    out["hybrid_top20"] = met
    print("hybrid", round(met["f1"], 4), round(met["fpr"], 4))
    save("unsw_features.json", out)


# --------------------------------------------------------------------------- #
def stage_tuning():
    Xtr, Xte, *_ = load()
    ytr = np.load(os.path.join(DATA, "unsw_y_train_bin.npy"))
    yte = np.load(os.path.join(DATA, "unsw_y_test_bin.npy"))
    rng = np.random.RandomState(RS)
    idx = rng.choice(len(Xtr), size=35000, replace=False)
    Xs, ys = Xtr.values[idx], ytr[idx]
    space = {"n_estimators": [100, 200],
             "max_depth": [15, 25, 40, None],
             "min_samples_leaf": [1, 2, 4],
             "min_samples_split": [2, 5, 10],
             "max_features": ["sqrt", 0.3, 0.5],
             "class_weight": [None, "balanced"]}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RS)
    search = RandomizedSearchCV(RandomForestClassifier(random_state=RS, n_jobs=2),
                                param_distributions=space, n_iter=10, scoring="f1",
                                cv=cv, random_state=RS, n_jobs=1, verbose=0, refit=False)
    t = time.time()
    search.fit(Xs, ys)
    out = {"tuning_time_s": time.time() - t,
           "best_params": {k: (v if not isinstance(v, (np.floating, float)) else round(float(v), 3))
                           for k, v in search.best_params_.items()},
           "best_cv_f1": float(search.best_score_),
           "table": [{**{k: (v if not isinstance(v, (np.floating, float)) else round(float(v), 3))
                         for k, v in p.items()},
                      "mean_f1": float(m), "std_f1": float(s)}
                     for p, m, s in zip(search.cv_results_["params"],
                                        search.cv_results_["mean_test_score"],
                                        search.cv_results_["std_test_score"])]}
    print("best", out["best_params"], round(out["best_cv_f1"], 4))
    save("unsw_tuning.json", out)

    best = dict(search.best_params_)
    t = time.time()
    rf = RandomForestClassifier(random_state=RS, n_jobs=2, **best)
    rf.fit(Xtr.values, ytr)
    out["final_train_time_s"] = time.time() - t
    out["final_test"] = bmetrics(yte, rf.predict(Xte.values), rf.predict_proba(Xte.values)[:, 1])
    out["final_train"] = None
    print("tuned test", {k: round(v, 4) for k, v in out["final_test"].items() if isinstance(v, float)})
    save("unsw_tuning.json", out)
    dump(rf, os.path.join(MODELS, "rf_tuned.joblib"), compress=0)


# --------------------------------------------------------------------------- #
def stage_multiclass():
    Xtr, Xte, *_ = load()
    ytr_cat = np.load(os.path.join(DATA, "unsw_y_train_cat.npy"), allow_pickle=True)
    yte_cat = np.load(os.path.join(DATA, "unsw_y_test_cat.npy"), allow_pickle=True)
    classes = sorted(pd.Series(ytr_cat).unique())
    rf = RandomForestClassifier(n_estimators=150, random_state=RS, n_jobs=1,
                                max_features="sqrt", class_weight="balanced_subsample")
    t = time.time()
    rf.fit(Xtr.values, ytr_cat)
    out = {"train_time_s": time.time() - t, "classes": classes}
    pred = rf.predict(Xte.values)
    out["accuracy"] = float(accuracy_score(yte_cat, pred))
    out["balanced_accuracy"] = float(balanced_accuracy_score(yte_cat, pred))
    out["macro_f1"] = float(f1_score(yte_cat, pred, average="macro"))
    out["weighted_f1"] = float(f1_score(yte_cat, pred, average="weighted"))
    out["mcc"] = float(matthews_corrcoef(yte_cat, pred))
    out["per_class"] = classification_report(yte_cat, pred, output_dict=True, zero_division=0)
    out["confusion"] = confusion_matrix(yte_cat, pred, labels=classes).tolist()
    out["support"] = {c: int((yte_cat == c).sum()) for c in classes}
    out["binary_from_multiclass"] = bmetrics((yte_cat != "Normal").astype(int),
                                             (pred != "Normal").astype(int),
                                             (pred != "Normal").astype(float))
    save("unsw_multiclass.json", out)
    print("multiclass acc", round(out["accuracy"], 4), "macroF1", round(out["macro_f1"], 4))
    dump(rf, os.path.join(MODELS, "rf_multiclass.joblib"), compress=0)


# --------------------------------------------------------------------------- #
def stage_timing():
    """Prediction-cost benchmarking with clean measurement (numpy arrays)."""
    Xtr, Xte, *_ = load()
    ytr = np.load(os.path.join(DATA, "unsw_y_train_bin.npy"))
    from joblib import load as jload
    out = {}
    feature_sets = {"full": None, "reduced": None}
    feats = json.load(open(os.path.join(RESULTS, "unsw_features.json")))
    top20 = feats["hybrid_top20"]["features"]

    configs = {
        "full_feature_200_trees": (Xte.values, dict(n_estimators=200)),
        "full_feature_100_trees": (Xte.values, dict(n_estimators=100)),
        "reduced_20_features_200_trees": (Xte[top20].values, dict(n_estimators=200)),
    }
    for name, (Xt, params) in configs.items():
        rf = RandomForestClassifier(random_state=RS, n_jobs=1, max_features="sqrt", **params)
        t = time.time()
        rf.fit(Xtr.values if Xt.shape[1] == Xtr.shape[1] else Xtr[top20].values, ytr)
        train_t = time.time() - t
        # warm-up
        rf.predict_proba(Xt[:100])
        batch = min(
            (lambda: (lambda s: (rf.predict(Xt), time.perf_counter() - s)[1])(time.perf_counter()))()
            for _ in range(3))
        n = len(Xt)
        # single-record calls (per-alert overhead a SOC would observe)
        reps = 300
        s = time.perf_counter()
        for i in range(reps):
            rf.predict(Xt[i:i + 1])
        single = (time.perf_counter() - s) / reps
        # micro-batches of 100
        s = time.perf_counter()
        for start in range(0, 20000, 100):
            rf.predict(Xt[start:start + 100])
        micro = (time.perf_counter() - s) / 20000
        out[name] = {"train_time_s": train_t,
                     "batch_total_s": batch,
                     "per_flow_us": batch / n * 1e6,
                     "throughput_flows_per_s": n / batch,
                     "single_record_latency_ms": single * 1e3,
                     "micro_batch_per_flow_ms": micro * 1e3,
                     "n_trees": int(rf.n_estimators),
                     "avg_tree_depth": float(np.mean([e.tree_.max_depth for e in rf.estimators_])),
                     "n_nodes": int(sum(e.tree_.node_count for e in rf.estimators_)),
                     "n_features": int(Xt.shape[1])}
        print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out[name].items()})
        save("unsw_timing.json", out)


# --------------------------------------------------------------------------- #
def stage_cv():
    Xtr, Xte, *_ = load()
    ytr = np.load(os.path.join(DATA, "unsw_y_train_bin.npy"))
    yte = np.load(os.path.join(DATA, "unsw_y_test_bin.npy"))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
    scoring = {"accuracy": "accuracy", "precision": "precision", "recall": "recall",
               "f1": "f1", "roc_auc": "roc_auc"}
    t = time.time()
    res = cross_validate(RandomForestClassifier(n_estimators=150, random_state=RS,
                                                n_jobs=2, max_features="sqrt"),
                         Xtr.values, ytr, cv=cv, scoring=scoring, n_jobs=1)
    out = {"cv_time_s": time.time() - t,
           "folds": {k: [float(x) for x in v] for k, v in res.items() if k.startswith("test_")},
           "summary": {k.replace("test_", ""): {"mean": float(np.mean(v)), "std": float(np.std(v))}
                       for k, v in res.items() if k.startswith("test_")}}
    # hold-out reference for the same configuration
    rf = RandomForestClassifier(n_estimators=150, random_state=RS, n_jobs=2, max_features="sqrt")
    rf.fit(Xtr.values, ytr)
    out["holdout"] = bmetrics(yte, rf.predict(Xte.values), rf.predict_proba(Xte.values)[:, 1])
    save("unsw_cv.json", out)
    print("CV", {k: round(v["mean"], 4) for k, v in out["summary"].items()})
    print("holdout", round(out["holdout"]["accuracy"], 4), round(out["holdout"]["f1"], 4))


# --------------------------------------------------------------------------- #
def stage_tree():
    """Single decision tree reference model (ensemble-value comparator)."""
    Xtr, Xte, *_ = load()
    ytr = np.load(os.path.join(DATA, "unsw_y_train_bin.npy"))
    yte = np.load(os.path.join(DATA, "unsw_y_test_bin.npy"))
    t = time.time()
    dt = DecisionTreeClassifier(random_state=RS)
    dt.fit(Xtr.values, ytr)
    met = bmetrics(yte, dt.predict(Xte.values), dt.predict_proba(Xte.values)[:, 1])
    met["train_time_s"] = time.time() - t
    met["n_nodes"] = int(dt.tree_.node_count)
    met["depth"] = int(dt.tree_.max_depth)
    save("unsw_tree.json", met)
    print("tree", {k: round(v, 4) for k, v in met.items() if isinstance(v, float)})


STAGES = {"base": stage_base, "imbalance": stage_imbalance, "features": stage_features,
          "tuning": stage_tuning, "multiclass": stage_multiclass, "timing": stage_timing,
          "cv": stage_cv, "tree": stage_tree}

if __name__ == "__main__":
    todo = sys.argv[1:] or ["base", "imbalance", "features", "tuning", "multiclass", "timing", "cv", "tree"]
    for s in todo:
        print(f"\n===== STAGE {s} =====", flush=True)
        try:
            STAGES[s]()
        except Exception as exc:
            print(f"STAGE {s} FAILED: {type(exc).__name__}: {exc}", flush=True)
