"""Generate all figures used in the research report."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(BASE, "analysis", "results")
F = os.path.join(BASE, "analysis", "figures")
os.makedirs(F, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "legend.frameon": False,
})
C = {"blue": "#1f4e79", "teal": "#2a9d8f", "orange": "#e76f51", "sand": "#e9c46a",
     "green": "#4c956c", "grey": "#6c757d", "red": "#b02a37", "purple": "#6a4c93"}


def j(name):
    with open(os.path.join(R, name)) as fh:
        return json.load(fh)


def save(fig, name):
    fig.savefig(os.path.join(F, name))
    plt.close(fig)
    print("wrote", name)


# --------------------------------------------------------------------------- #
def fig_pipeline():
    """Three-stage analysis workflow (data -> modelling -> interpretation)."""
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    ax.set_xlim(0, 12.4); ax.set_ylim(0, 10); ax.axis("off"); ax.grid(False)

    colx = {0: 0.20, 1: 4.50, 2: 8.72}
    w, h = 3.25, 1.15
    boxes = [
        (0, 8.35, "Benchmark flow data\n(UNSW-NB15, CICIDS2017)", C["blue"]),
        (0, 6.45, "Data audit and cleaning\n(duplicates, \u00b1inf)", C["grey"]),
        (0, 4.55, "Pre-processing\n(encoding, rare levels)", C["grey"]),
        (0, 2.65, "Stratified partitions\n(train / hold-out)", C["grey"]),
        (1, 8.35, "Feature selection\n(importance, MI, r-pruning)", C["teal"]),
        (1, 6.45, "Random Forest training\n(4 configurations)", C["teal"]),
        (1, 4.55, "Multiclass extension\n(10 categories)", C["teal"]),
        (1, 2.65, "Evaluation on\nheld-out flows", C["purple"]),
        (2, 5.35, "Interpretation\ndetection quality \u00b7 operational\ncost \u00b7 benchmark validity", C["orange"], 1.6),
    ]
    for box in boxes:
        c, y, txt, col = box[:4]
        bh = box[4] if len(box) > 4 else h
        ax.add_patch(FancyBboxPatch((colx[c], y), w, bh,
                                    boxstyle="round,pad=0.05,rounding_size=0.10",
                                    linewidth=1.1, edgecolor=col,
                                    facecolor=matplotlib.colors.to_rgba(col, 0.10)))
        ax.text(colx[c] + w / 2, y + bh / 2, txt, ha="center", va="center",
                fontsize=6.7, linespacing=1.5, color="#1a1a1a")

    def arrow(p1, p2, style="-|>", ls="-"):
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=9,
                                     linewidth=0.9, color="#495057", linestyle=ls,
                                     shrinkA=0, shrinkB=0))

    cy = lambda c, y: (colx[c] + w / 2, y)
    for y0, y1 in [(8.35, 7.60), (6.45, 5.70), (4.55, 3.80)]:
        arrow(cy(0, y0), cy(0, y1))
    # data -> modelling entry
    arrow((colx[0] + w, 3.22), (colx[1], 3.22))
    arrow((colx[1] + 0.55, 3.80), (colx[1] + 0.55, 4.55), ls="--")
    # vertical chain in modelling column
    for y0, y1 in [(8.35, 7.60), (6.45, 5.70), (4.55, 3.80)]:
        arrow(cy(1, y0), cy(1, y1))
    # evaluation -> interpretation
    arrow((colx[1] + w, 3.22), (colx[2], 3.22))
    arrow((colx[2] + 0.55, 3.22), (colx[2] + 0.55, 5.30))

    ax.text(6.0, 1.35,
            "Configurations: baseline  |  class-weighted  |  SMOTE-balanced  |  hyper-parameter tuned",
            ha="center", fontsize=6.6, color="#1a1a1a")
    ax.text(6.0, 0.80,
            "Metrics: accuracy, precision, recall, F1, ROC-AUC, false positive rate, MCC, latency",
            ha="center", fontsize=6.6, color="#1a1a1a")
    ax.text(6.0, 0.22, "All runs seeded (random_state = 42) for reproducibility",
            ha="center", fontsize=6.4, style="italic", color=C["grey"])
    save(fig, "fig3_1_pipeline.png")


def fig_distributions():
    a = j("audit_unsw.json")
    cats = a["attack_cat_train"]
    order = sorted(cats, key=lambda k: -cats[k])
    tr_raw = [cats[c] for c in order]
    tr_clean = [a["attack_cat_train_clean"].get(c, 0) for c in order]
    te_clean = [a["attack_cat_test_clean"].get(c, 0) for c in order]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    x = np.arange(len(order)); w = 0.27
    ax = axes[0]
    ax.bar(x - w, tr_raw, w, label="Train (raw)", color=C["grey"], alpha=0.85)
    ax.bar(x, tr_clean, w, label="Train (de-duplicated)", color=C["blue"])
    ax.bar(x + w, te_clean, w, label="Test (de-duplicated)", color=C["teal"])
    ax.set_yscale("log"); ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("Flow records (log scale)")
    ax.set_title("(a) Category composition and duplication")
    ax.legend(fontsize=6.5, loc="upper right")
    ax = axes[1]
    loss = [100 * (1 - c / r) for r, c in zip(tr_raw, tr_clean)]
    bars = ax.barh(order[::-1], loss[::-1], color=C["orange"])
    ax.set_xlabel("Share of records removed as duplicates (%)")
    ax.set_title("(b) Duplication intensity by category")
    for b, v in zip(bars, loss[::-1]):
        ax.text(v + 1, b.get_y() + b.get_height() / 2, f"{v:.0f}%", va="center", fontsize=6.5)
    ax.set_xlim(0, 100)
    for a_ in axes:
        a_.tick_params(labelsize=7)
    fig.tight_layout()
    save(fig, "fig4_1_distribution.png")


def fig_confusion():
    b = j("unsw_base.json")
    t = j("unsw_tuning.json")["final_test"]
    cm = np.array([[t["tn"], t["fp"]], [t["fn"], t["tp"]]])
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8))
    labels = ["Benign (0)", "Attack (1)"]
    for ax, data, title, fmt in [
            (axes[0], cm, "(a) Counts", "d"),
            (axes[1], cm / cm.sum(axis=1, keepdims=True) * 100, "(b) Row-normalised (%)", ".1f")]:
        im = ax.imshow(data, cmap="Blues", vmin=0)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(labels); ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(title); ax.grid(False)
        for i in range(2):
            for k in range(2):
                v = data[i, k]
                txt = f"{int(v)}" if fmt == "d" else f"{v:.1f}%"
                ax.text(k, i, txt, ha="center", va="center", fontsize=9,
                        color="white" if v > data.max() * 0.5 else "#1a1a1a")
    fig.tight_layout()
    save(fig, "fig4_2_confusion_binary.png")


def fig_roc_pr():
    b = j("unsw_base.json")
    t = j("unsw_tuning.json")["final_test"]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9))
    ax = axes[0]
    ax.plot(b["roc"]["fpr"], b["roc"]["tpr"], color=C["blue"], lw=1.6,
            label=f"Random Forest (AUC = {t['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color=C["grey"], lw=1, label="No-skill baseline")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("(a) ROC curve"); ax.legend(fontsize=7, loc="lower right")
    ax.set_xlim(-0.01, 1.0); ax.set_ylim(0, 1.01)
    ax = axes[1]
    ax.plot(b["pr"]["recall"], b["pr"]["precision"], color=C["teal"], lw=1.6,
            label=f"Random Forest (AP = {t['pr_auc']:.3f})")
    base = np.mean(np.load(os.path.join(BASE, "data", "unsw_y_test_bin.npy")))
    ax.axhline(base, ls="--", color=C["grey"], lw=1, label=f"Chance ({base:.2f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("(b) Precision–recall curve"); ax.legend(fontsize=7, loc="lower left")
    ax.set_xlim(0, 1.0); ax.set_ylim(0.4, 1.02)
    fig.tight_layout()
    save(fig, "fig4_3_roc_pr.png")


def fig_importance():
    b = j("unsw_base.json")
    imp = b["feature_importance"][:20]
    names = [d["feature"] for d in imp][::-1]
    vals = [d["importance"] for d in imp][::-1]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    ax.barh(names, vals, color=C["blue"])
    ax.set_xlabel("Mean decrease in impurity")
    ax.set_title("(a) Twenty most important flow features")
    ax.tick_params(labelsize=7)
    ax = axes[1]
    full = b["feature_importance"]
    cum = np.cumsum([d["importance"] for d in full])
    ax.plot(range(1, len(cum) + 1), cum, color=C["teal"], lw=1.8)
    ax.axhline(0.95, ls="--", color=C["orange"], lw=1, label="95% of total importance")
    n95 = int(np.argmax(cum >= 0.95) + 1)
    ax.axvline(n95, ls=":", color=C["orange"], lw=1)
    ax.annotate(f"{n95} features", xy=(n95, 0.55), xytext=(n95 + 6, 0.62), fontsize=7,
                arrowprops=dict(arrowstyle="->", lw=0.8, color=C["grey"]))
    ax.set_xlabel("Number of features retained"); ax.set_ylabel("Cumulative importance")
    ax.set_title("(b) Cumulative importance")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    save(fig, "fig4_4_importance.png")


def fig_feature_selection():
    fs = j("unsw_features.json")
    imp = fs["importance_selection"]
    mi = fs["mi_selection"]
    ks = sorted(int(k) for k in imp)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    ax = axes[0]
    ax.plot(ks, [imp[str(k)]["f1"] for k in sorted(int(x) for x in imp)], "o-",
            color=C["blue"], label="RF importance")
    ax.plot(sorted(int(k) for k in mi), [mi[str(k)]["f1"] for k in sorted(int(x) for x in mi)],
            "s--", color=C["teal"], label="Mutual information")
    base_f1 = j("unsw_base.json")["test"]["f1"]
    ax.axhline(base_f1, color=C["orange"], ls=":", lw=1.2, label=f"Full feature set ({base_f1:.3f})")
    ax.set_xlabel("Features retained"); ax.set_ylabel("F1-score (attack class)")
    ax.set_title("(a) Detection performance vs. dimensionality")
    ax.legend(fontsize=6.5, loc="lower right"); ax.set_ylim(0.75, 0.84)
    ax = axes[1]
    ax2 = ax.twinx()
    ax.plot(ks, [imp[str(k)]["train_time_s"] for k in sorted(int(x) for x in imp)], "o-",
            color=C["purple"], label="Training time")
    ax.set_ylabel("Training time (s)", color=C["purple"])
    ax.set_xlabel("Features retained")
    ax2.plot(ks, [100 * imp[str(k)]["fpr"] for k in sorted(int(x) for x in imp)], "s--",
             color=C["red"], label="False positive rate")
    ax2.set_ylabel("False positive rate (%)", color=C["red"])
    ax2.grid(False); ax2.spines["right"].set_visible(True)
    ax.set_title("(b) Cost and false-alarm trade-off")
    fig.tight_layout()
    save(fig, "fig4_5_feature_selection.png")


def fig_variants():
    b = j("unsw_base.json")["test"]
    imb = j("unsw_imbalance.json")
    tun = j("unsw_tuning.json")["final_test"]
    fs = j("unsw_features.json")
    tree = j("unsw_tree.json")
    variants = [
        ("Baseline RF\n(73 features)", b),
        ("Class-balanced\nRF", imb["balanced"]),
        ("SMOTE +\nRF", imb["smote"]),
        ("Tuned RF\n(CV search)", tun),
        ("RF, 20 features\n(hybrid)", fs["hybrid_top20"]),
        ("Single decision\ntree", tree),
    ]
    names = [v[0] for v in variants]
    metrics = [("f1", "F1-score", C["blue"]), ("recall", "Recall", C["teal"]),
               ("fpr", "False positive rate", C["orange"])]
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    x = np.arange(len(variants)); w = 0.26
    for i, (key, lab, col) in enumerate(metrics):
        vals = [v[1][key] for v in variants]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=lab, color=col)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v:.2f}",
                    ha="center", fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=7)
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.1)
    ax.legend(fontsize=7, ncol=3, loc="upper center")
    ax.set_title("Detection performance of the experimental configurations (UNSW-NB15 hold-out)")
    save(fig, "fig4_6_variants.png")


def fig_threshold():
    b = j("unsw_base.json")["threshold_sweep"]
    th = [s["threshold"] for s in b]
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    ax.plot(th, [s["precision"] for s in b], "-", color=C["blue"], label="Precision")
    ax.plot(th, [s["recall"] for s in b], "-", color=C["teal"], label="Recall (TPR)")
    ax.plot(th, [s["f1"] for s in b], "-", color=C["green"], label="F1-score")
    ax.plot(th, [s["fpr"] for s in b], "--", color=C["orange"], label="False positive rate")
    ax.axvline(0.5, color=C["grey"], ls=":", lw=1.1)
    ax.text(0.505, 0.03, "default 0.50", fontsize=6.5, color=C["grey"])
    best = max(b, key=lambda s: s["f1"])
    ax.axvline(best["threshold"], color=C["red"], ls=":", lw=1.1)
    ax.annotate(f"F1-optimal ({best['threshold']:.2f})\nF1 = {best['f1']:.3f}, FPR = {best['fpr']*100:.1f}%",
                xy=(best["threshold"], best["f1"]), xytext=(best["threshold"] - 0.37, 0.35),
                fontsize=6.5, arrowprops=dict(arrowstyle="->", lw=0.8, color=C["grey"]))
    ax.set_xlabel("Decision threshold on the attack probability")
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, ncol=2, loc="lower left")
    ax.set_title("Operating-point trade-off for the tuned Random Forest")
    save(fig, "fig4_7_threshold.png")


def fig_per_class():
    m = j("unsw_multiclass.json")
    pc = m["per_class"]
    classes = [c for c in m["classes"]]
    classes = sorted(classes, key=lambda c: -m["support"][c])
    prec = [pc[c]["precision"] for c in classes]
    rec = [pc[c]["recall"] for c in classes]
    f1 = [pc[c]["f1-score"] for c in classes]
    sup = [m["support"][c] for c in classes]
    fig, ax = plt.subplots(figsize=(7.0, 3.1))
    x = np.arange(len(classes)); w = 0.26
    ax.bar(x - w, prec, w, label="Precision", color=C["blue"])
    ax.bar(x, rec, w, label="Recall", color=C["teal"])
    ax.bar(x + w, f1, w, label="F1-score", color=C["orange"])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n(n={n:,})" for c, n in zip(classes, sup)],
                       rotation=30, ha="right", fontsize=6.5, linespacing=1.4)
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.06)
    ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.03))
    ax.set_ylim(0, 1.28)
    ax.set_title("Per-category performance of the multiclass Random Forest (UNSW-NB15 hold-out)")
    save(fig, "fig4_8_per_class.png")


def fig_multiclass_cm():
    m = j("unsw_multiclass.json")
    cm = np.array(m["confusion"], dtype=float)
    classes = m["classes"]
    norm = cm / cm.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=40, ha="right", fontsize=7)
    ax.set_yticklabels(classes, fontsize=7)
    ax.set_xlabel("Predicted category"); ax.set_ylabel("Actual category")
    ax.grid(False)
    for i in range(len(classes)):
        for k in range(len(classes)):
            v = norm[i, k]
            if v >= 1:
                ax.text(k, i, f"{v:.0f}", ha="center", va="center", fontsize=5.6,
                        color="white" if v > 50 else "#1a1a1a")
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("Row percentage (%)", fontsize=7)
    ax.set_title("Normalised confusion matrix: attack-category classification")
    save(fig, "fig4_9_multiclass_cm.png")


def fig_timing():
    t = j("unsw_timing.json")
    labels = ["Full features\n200 trees", "Full features\n100 trees", "20 features\n200 trees"]
    keys = ["full_feature_200_trees", "full_feature_100_trees", "reduced_20_features_200_trees"]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))
    ax = axes[0]
    vals = [t[k]["throughput_flows_per_s"] / 1000 for k in keys]
    bars = ax.bar(labels, vals, color=[C["blue"], C["teal"], C["purple"]], width=0.6)
    for b_, v in zip(bars, vals):
        ax.text(b_.get_x() + b_.get_width() / 2, v + 2, f"{v:.0f}k", ha="center", fontsize=7)
    ax.set_ylabel("Throughput (thousand flows/s)")
    ax.set_title("(a) Batch throughput"); ax.tick_params(labelsize=6.5)
    ax = axes[1]
    vals = [t[k]["micro_batch_per_flow_ms"] for k in keys]
    bars = ax.bar(labels, vals, color=[C["blue"], C["teal"], C["purple"]], width=0.6)
    for b_, v in zip(bars, vals):
        ax.text(b_.get_x() + b_.get_width() / 2, v + 0.004, f"{v:.3f}", ha="center", fontsize=7)
    ax.set_ylabel("Latency per flow (ms)")
    ax.set_title("(b) Micro-batch latency (100 flows)"); ax.tick_params(labelsize=6.5)
    ax = axes[2]
    vals = [t[k]["single_record_latency_ms"] for k in keys]
    bars = ax.bar(labels, vals, color=[C["blue"], C["teal"], C["purple"]], width=0.6)
    for b_, v in zip(bars, vals):
        ax.text(b_.get_x() + b_.get_width() / 2, v + 0.2, f"{v:.1f}", ha="center", fontsize=7)
    ax.set_ylabel("Latency per call (ms)")
    ax.set_title("(c) Single-record API call"); ax.tick_params(labelsize=6.5)
    fig.tight_layout()
    save(fig, "fig4_10_timing.png")


def fig_cv():
    cv = j("unsw_cv.json")
    folds = cv["folds"]
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    metrics = [("test_accuracy", "Accuracy"), ("test_precision", "Precision"),
               ("test_recall", "Recall"), ("test_f1", "F1-score")]
    positions = np.arange(len(metrics))
    for i, (key, lab) in enumerate(metrics):
        vals = folds[key]
        ax.scatter([i] * len(vals), vals, color=C["teal"], s=16, alpha=0.85,
                   label="Five CV folds" if i == 0 else None, zorder=3)
        ax.plot([i - 0.22, i + 0.22], [np.mean(vals)] * 2, color=C["teal"], lw=2)
    hold = [cv["holdout"]["accuracy"], cv["holdout"]["precision"],
            cv["holdout"]["recall"], cv["holdout"]["f1"]]
    ax.scatter(positions, hold, marker="D", color=C["orange"], s=34, zorder=4,
               label="Hold-out partition (unique records)")
    for i, v in enumerate(hold):
        ax.annotate(f"{v:.3f}", xy=(i, v), xytext=(i + 0.09, v - 0.055), fontsize=6.5,
                    color=C["orange"])
    ax.set_xticks(positions); ax.set_xticklabels([m[1] for m in metrics])
    ax.set_ylabel("Score"); ax.set_ylim(0.72, 1.0)
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title("Cross-validated estimates versus held-out performance")
    save(fig, "fig4_11_cv.png")


def fig_cicids():
    c = j("cicids_results.json")
    u = j("unsw_tuning.json")["final_test"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    counts = pd.Series(c["label_counts"]).sort_values()
    import re
    labels = [re.sub(r"Web_Attack_.?_?", "Web: ", str(i)).replace("_", " ")
              for i in counts.index]
    ax.barh(labels[::-1], counts.values[::-1], color=C["teal"])
    ax.set_xscale("log")
    ax.set_xlabel("Flow records (log scale)")
    ax.set_title("(a) CICIDS2017 sample composition")
    ax.tick_params(labelsize=6.2)
    ax = axes[1]
    metrics = [("accuracy", "Accuracy"), ("recall", "Recall"), ("precision", "Precision"),
               ("f1", "F1-score"), ("fpr", "False positive rate")]
    x = np.arange(len(metrics)); w = 0.36
    uv = [u.get(k) for k, _ in metrics]
    cv_ = [c["baseline"].get(k if k != "fpr" else "fpr") for k, _ in metrics]
    ax.bar(x - w / 2, uv, w, label="UNSW-NB15", color=C["blue"])
    ax.bar(x + w / 2, cv_, w, label="CICIDS2017", color=C["teal"])
    for i, (a_, b_) in enumerate(zip(uv, cv_)):
        ax.text(i - w / 2, a_ + 0.01, f"{a_:.2f}", ha="center", fontsize=6)
        ax.text(i + w / 2, b_ + 0.01, f"{b_:.3f}" if b_ < 0.01 else f"{b_:.2f}",
                ha="center", fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels([m[1] for m in metrics], rotation=20, ha="right", fontsize=7)
    ax.set_ylim(0, 1.42); ax.set_ylabel("Score")
    ax.legend(fontsize=6.6, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    ax.set_title("(b) Same pipeline, two benchmarks")
    fig.tight_layout()
    save(fig, "fig4_12_cicids.png")


def fig_dedup():
    ref = j("reference_pipeline.json")
    base = j("unsw_base.json")
    groups = [("Literature-standard\n(raw train → raw test)", ref["raw_pipeline_raw_test"]),
              ("Raw-trained model\n→ de-duplicated test", ref["raw_pipeline_dedup_test"]),
              ("De-duplicated\npipeline (this study)", base["test"])]
    metrics = [("accuracy", "Accuracy", C["blue"]), ("f1", "F1-score", C["teal"]),
               ("fpr", "False positive rate", C["orange"])]
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    x = np.arange(len(groups)); w = 0.26
    for i, (key, lab, col) in enumerate(metrics):
        vals = [
            g[1].get(key, g[1].get("fpr")) if key != "fpr" else g[1]["fpr"] for g in groups]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=lab, color=col)
        for b_, v in zip(bars, vals):
            ax.text(b_.get_x() + b_.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels([g[0] for g in groups], fontsize=7)
    ax.set_ylim(0, 1.08); ax.set_ylabel("Score")
    ax.legend(fontsize=7, ncol=3, loc="upper right")
    ax.set_title("Effect of duplicate records on reported performance")
    save(fig, "fig4_13_dedup.png")


if __name__ == "__main__":
    fig_pipeline()
    fig_distributions()
    fig_confusion()
    fig_roc_pr()
    fig_importance()
    fig_feature_selection()
    fig_variants()
    fig_threshold()
    fig_per_class()
    fig_multiclass_cm()
    fig_timing()
    fig_cv()
    fig_cicids()
    fig_dedup()
