import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_curve,
)

ROOT = "/home/shanyiye/CLAM"
LABEL_CSV = os.path.join(ROOT, "dataset_csv", "gcb_vs_nongcb.csv")

EVAL_DIRS = {
    "CLAM_SB_WS": os.path.join(ROOT, "eval_results", "EVAL_dlbcl_gcb_nongcb_clam_sb_eval"),
    "CLAM_SB_noWS": os.path.join(ROOT, "eval_results", "EVAL_dlbcl_gcb_nongcb_clam_sb_nows_eval"),
    "MIL": os.path.join(ROOT, "eval_results", "EVAL_dlbcl_gcb_nongcb_mil_eval"),
    "CLAM_MB_WS": os.path.join(ROOT, "eval_results", "EVAL_dlbcl_gcb_nongcb_clam_mb_eval"),
    "CLAM_MB_noWS": os.path.join(ROOT, "eval_results", "EVAL_dlbcl_gcb_nongcb_clam_mb_nows_eval"),
}

OUT_ROOT = os.path.join(ROOT, "analysis_results")
os.makedirs(OUT_ROOT, exist_ok=True)


def norm_slide(x):
    x = str(x).strip()
    x = re.sub(r"\.(pt|h5|sdpc|svs)$", "", x, flags=re.I)
    return x


def norm_label(x):
    s = str(x).strip().lower()
    if s in {"0", "gcb"}:
        return 0
    if s in {"1", "non-gcb", "nongcb", "non_gcb", "abc"}:
        return 1
    raise ValueError(f"unknown label: {x}")


def first_existing(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None


def load_mapping():
    df = pd.read_csv(LABEL_CSV)
    need = {"case_id", "slide_id", "label"}
    miss = need - set(df.columns)
    if miss:
        raise RuntimeError(f"label csv missing columns: {miss}")

    df = df[["case_id", "slide_id", "label"]].copy()
    df["slide_id"] = df["slide_id"].map(norm_slide)
    df["label_num"] = df["label"].map(norm_label)
    return df


def load_eval_predictions(eval_dir):
    files = sorted(glob.glob(os.path.join(eval_dir, "fold_*.csv")))
    if not files:
        raise RuntimeError(f"no fold_*.csv found in {eval_dir}")

    parts = []
    for fp in files:
        df = pd.read_csv(fp)

        slide_col = first_existing(df.columns, ["slide_id", "Unnamed: 0"])
        if slide_col is None:
            raise RuntimeError(f"{fp}: cannot find slide_id column")

        true_col = first_existing(df.columns, ["Y", "y_true", "label", "target"])
        pred_col = first_existing(df.columns, ["Y_hat", "y_pred", "pred", "prediction"])
        prob_col = first_existing(df.columns, ["p_1", "prob_1", "score_1", "positive_prob"])

        if prob_col is None:
            p_cols = [c for c in df.columns if str(c).startswith("p_")]
            if "p_1" in p_cols:
                prob_col = "p_1"
            elif len(p_cols) >= 2:
                prob_col = sorted(p_cols)[-1]

        if prob_col is None:
            raise RuntimeError(f"{fp}: cannot find positive-class probability column")

        fold_match = re.search(r"fold_(\d+)\.csv", os.path.basename(fp))
        fold_id = int(fold_match.group(1)) if fold_match else -1

        out = pd.DataFrame({
            "slide_id": df[slide_col].map(norm_slide),
            "prob_1": df[prob_col].astype(float),
            "fold": fold_id,
        })

        if true_col is not None:
            out["y_true"] = df[true_col].astype(int)

        if pred_col is not None:
            out["y_pred"] = df[pred_col].astype(int)

        parts.append(out)

    pred = pd.concat(parts, ignore_index=True)

    if "y_pred" not in pred.columns:
        pred["y_pred"] = (pred["prob_1"] >= 0.5).astype(int)

    return pred


def calc_metrics(y_true, y_pred, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    return {
        "auc": auc,
        "acc": acc,
        "f1": f1,
        "sensitivity": sens,
        "specificity": spec,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def save_roc(y_true, y_prob, out_png, title):
    if len(np.unique(y_true)) < 2:
        return
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def save_cm(cm, out_png, title):
    labels = ["GCB", "non-GCB"]
    plt.figure(figsize=(5, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.xticks([0, 1], labels)
    plt.yticks([0, 1], labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def load_summary(eval_dir):
    files = sorted(glob.glob(os.path.join(eval_dir, "summary*.csv")))
    if not files:
        raise RuntimeError(f"no summary*.csv found in {eval_dir}")
    return pd.read_csv(files[0])


def main():
    mapping = load_mapping()
    rows = []

    for exp_name, eval_dir in EVAL_DIRS.items():
        print(f"\n=== processing {exp_name} ===")
        out_dir = os.path.join(OUT_ROOT, exp_name)
        os.makedirs(out_dir, exist_ok=True)

        summary_df = load_summary(eval_dir)
        pred_df = load_eval_predictions(eval_dir)

        merged = pred_df.merge(
            mapping[["case_id", "slide_id", "label", "label_num"]],
            on="slide_id",
            how="left"
        )

        missing = merged["case_id"].isna().sum()
        print(f"missing case_id rows: {missing}")
        if missing > 0:
            merged[merged["case_id"].isna()].to_csv(
                os.path.join(out_dir, "unmatched_slides.csv"),
                index=False
            )

        merged = merged.dropna(subset=["case_id", "label_num"]).copy()

        if "y_true" not in merged.columns or merged["y_true"].isna().all():
            merged["y_true"] = merged["label_num"].astype(int)
        else:
            merged["y_true"] = merged["y_true"].fillna(merged["label_num"]).astype(int)

        merged["y_pred"] = merged["y_pred"].astype(int)
        merged["distance_to_05"] = (merged["prob_1"] - 0.5).abs()

        slide_metrics = calc_metrics(
            merged["y_true"],
            merged["y_pred"],
            merged["prob_1"]
        )

        case_df = merged.groupby("case_id").agg(
            prob_1=("prob_1", "mean"),
            y_true=("label_num", "first"),
            n_slides=("slide_id", "nunique")
        ).reset_index()

        case_df["y_pred"] = (case_df["prob_1"] >= 0.5).astype(int)
        case_df["distance_to_05"] = (case_df["prob_1"] - 0.5).abs()

        case_metrics = calc_metrics(
            case_df["y_true"],
            case_df["y_pred"],
            case_df["prob_1"]
        )

        merged.to_csv(os.path.join(out_dir, "all_slide_predictions.csv"), index=False)
        case_df.to_csv(os.path.join(out_dir, "all_case_predictions.csv"), index=False)

        merged[merged["y_true"] != merged["y_pred"]].sort_values(
            "distance_to_05", ascending=False
        ).to_csv(os.path.join(out_dir, "wrong_slides.csv"), index=False)

        case_df[case_df["y_true"] != case_df["y_pred"]].sort_values(
            "distance_to_05", ascending=False
        ).to_csv(os.path.join(out_dir, "wrong_cases.csv"), index=False)

        case_df.sort_values("distance_to_05", ascending=True).head(10).to_csv(
            os.path.join(out_dir, "borderline_cases.csv"),
            index=False
        )

        slide_cm = confusion_matrix(merged["y_true"], merged["y_pred"], labels=[0, 1])
        case_cm = confusion_matrix(case_df["y_true"], case_df["y_pred"], labels=[0, 1])

        save_roc(merged["y_true"], merged["prob_1"],
                 os.path.join(out_dir, "slide_roc.png"),
                 f"{exp_name} Slide-level ROC")

        save_roc(case_df["y_true"], case_df["prob_1"],
                 os.path.join(out_dir, "case_roc.png"),
                 f"{exp_name} Case-level ROC")

        save_cm(slide_cm,
                os.path.join(out_dir, "slide_confusion_matrix.png"),
                f"{exp_name} Slide-level CM")

        save_cm(case_cm,
                os.path.join(out_dir, "case_confusion_matrix.png"),
                f"{exp_name} Case-level CM")

        report_lines = [
            f"Experiment: {exp_name}",
            "",
            "Fold summary from eval summary.csv",
            f"mean_test_auc = {summary_df['test_auc'].mean():.4f}",
            f"std_test_auc  = {summary_df['test_auc'].std(ddof=1):.4f}",
            f"mean_test_acc = {summary_df['test_acc'].mean():.4f}",
            f"std_test_acc  = {summary_df['test_acc'].std(ddof=1):.4f}",
            "",
            "Overall slide-level out-of-fold metrics",
            f"auc         = {slide_metrics['auc']:.4f}",
            f"acc         = {slide_metrics['acc']:.4f}",
            f"f1          = {slide_metrics['f1']:.4f}",
            f"sensitivity = {slide_metrics['sensitivity']:.4f}",
            f"specificity = {slide_metrics['specificity']:.4f}",
            f"tn={slide_metrics['tn']}, fp={slide_metrics['fp']}, fn={slide_metrics['fn']}, tp={slide_metrics['tp']}",
            "",
            "Case-level metrics",
            f"auc         = {case_metrics['auc']:.4f}",
            f"acc         = {case_metrics['acc']:.4f}",
            f"f1          = {case_metrics['f1']:.4f}",
            f"sensitivity = {case_metrics['sensitivity']:.4f}",
            f"specificity = {case_metrics['specificity']:.4f}",
            f"tn={case_metrics['tn']}, fp={case_metrics['fp']}, fn={case_metrics['fn']}, tp={case_metrics['tp']}",
        ]

        with open(os.path.join(out_dir, "report.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))

        rows.append({
            "model": exp_name,
            "fold_mean_test_auc": summary_df["test_auc"].mean(),
            "fold_std_test_auc": summary_df["test_auc"].std(ddof=1),
            "fold_mean_test_acc": summary_df["test_acc"].mean(),
            "fold_std_test_acc": summary_df["test_acc"].std(ddof=1),
            "slide_auc_oof": slide_metrics["auc"],
            "slide_acc_oof": slide_metrics["acc"],
            "slide_f1_oof": slide_metrics["f1"],
            "slide_sensitivity_oof": slide_metrics["sensitivity"],
            "slide_specificity_oof": slide_metrics["specificity"],
            "case_auc": case_metrics["auc"],
            "case_acc": case_metrics["acc"],
            "case_f1": case_metrics["f1"],
            "case_sensitivity": case_metrics["sensitivity"],
            "case_specificity": case_metrics["specificity"],
            "n_slides": len(merged),
            "n_cases": len(case_df),
        })

    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(os.path.join(OUT_ROOT, "model_comparison.csv"), index=False)
    print("\nSaved:", os.path.join(OUT_ROOT, "model_comparison.csv"))


if __name__ == "__main__":
    main()
