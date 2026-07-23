"""Mid-semester at-risk model: selection, threshold choice, evaluation.

Protocol (all decisions made on training data only):
1. Stratified 80/20 holdout split.
2. 5-fold stratified CV on the training split to compare logistic regression
   vs. gradient boosting (ROC AUC).
3. Operating threshold chosen from the winning model's out-of-fold training
   probabilities: the highest threshold whose OOF recall still meets
   RECALL_TARGET (missing an at-risk student costs more than a false alert).
4. Final numbers reported once, on the untouched test split.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from . import config, viz
from .features import build_preprocessor, feature_names, make_xy


def candidate_models() -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline([
            ("prep", build_preprocessor(scale_numeric=True)),
            ("clf", LogisticRegression(max_iter=2000, random_state=config.SEED)),
        ]),
        "gradient_boosting": Pipeline([
            ("prep", build_preprocessor(scale_numeric=False)),
            ("clf", HistGradientBoostingClassifier(random_state=config.SEED)),
        ]),
    }


@dataclass
class ModelResult:
    name: str
    cv_auc_mean: float
    cv_auc_std: float


def _lr_pipe(numeric: list[str], categorical: list[str]) -> Pipeline:
    return Pipeline([
        ("prep", build_preprocessor(scale_numeric=True,
                                    numeric=numeric, categorical=categorical)),
        ("clf", LogisticRegression(max_iter=2000, random_state=config.SEED)),
    ])


def ablation_table(X_train, y_train, cv) -> tuple[float, pd.DataFrame]:
    """Feature-group ablation: CV AUC with each group alone and with it removed,
    against a baseline using every candidate column (model inputs +
    demographics). Training split only; retrains from scratch each time, so
    within-group correlation can't hide a group's contribution the way it can
    with per-column permutation importance."""
    all_cols = [c for cols in config.FEATURE_GROUPS.values() for c in cols]
    numeric_all = config.NUMERIC_FEATURES + config.DEMOGRAPHIC_NUMERIC
    cat_all = config.CATEGORICAL_FEATURES + config.DEMOGRAPHIC_CATEGORICAL

    def cv_auc(cols: list[str]) -> float:
        pipe = _lr_pipe([c for c in numeric_all if c in cols],
                        [c for c in cat_all if c in cols])
        return cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc").mean()

    baseline = cv_auc(all_cols)
    rows = [{
        "group": name,
        "CV AUC (group alone)": cv_auc(feats),
        "CV AUC (group removed)": cv_auc([c for c in all_cols if c not in feats]),
    } for name, feats in config.FEATURE_GROUPS.items()]
    table = pd.DataFrame(rows)
    table["AUC cost of removal"] = baseline - table["CV AUC (group removed)"]
    return baseline, table.sort_values("AUC cost of removal", ascending=False).round(3)


def pick_threshold(y_true: np.ndarray, proba: np.ndarray, recall_target: float) -> float:
    """Highest threshold whose recall still meets the target (max precision
    subject to the recall constraint, since precision rises with threshold)."""
    order = np.sort(np.unique(proba))[::-1]
    positives = y_true.sum()
    for t in order:
        recall = ((proba >= t) & (y_true == 1)).sum() / positives
        if recall >= recall_target:
            return float(t)
    return float(order[-1])


def _threshold_curve(y_true: np.ndarray, proba: np.ndarray) -> pd.DataFrame:
    rows = []
    # cap at 0.90: above that almost nothing is flagged and precision under
    # zero_division=0 produces a misleading cliff
    for t in np.arange(0.05, 0.901, 0.01):
        pred = (proba >= t).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, pred, average="binary", zero_division=0
        )
        rows.append({"threshold": t, "precision": p, "recall": r, "f1": f1})
    return pd.DataFrame(rows)


def subgroup_table(df_test: pd.DataFrame, y_true: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    """Per-group operating metrics for the shipped model (manual computation;
    Fairlearn's MetricFrame is the production-grade equivalent)."""
    rows = []
    frame = df_test.assign(_y=y_true, _pred=pred)
    for attr in config.SENSITIVE_ATTRS:
        for value, g in frame.groupby(attr, dropna=False):
            tp = ((g._pred == 1) & (g._y == 1)).sum()
            fp = ((g._pred == 1) & (g._y == 0)).sum()
            fn = ((g._pred == 0) & (g._y == 1)).sum()
            tn = ((g._pred == 0) & (g._y == 0)).sum()
            rows.append({
                "attribute": attr,
                "group": str(value),
                "n": len(g),
                "base_rate": g._y.mean(),
                "flag_rate": g._pred.mean(),
                "recall_tpr": tp / (tp + fn) if tp + fn else np.nan,
                "fpr": fp / (fp + tn) if fp + tn else np.nan,
            })
    return pd.DataFrame(rows).round(3)


def _coefficient_table(pipe: Pipeline) -> pd.DataFrame | None:
    """Standardized coefficients when the selected model is logistic regression
    (on scaled inputs these are directly comparable across features)."""
    clf = pipe.named_steps["clf"]
    if not isinstance(clf, LogisticRegression):
        return None
    names = feature_names(pipe.named_steps["prep"])
    coefs = pd.Series(clf.coef_[0], index=names, name="coef")
    top = coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(12)
    return top.round(3).to_frame()


def run(primary: pd.DataFrame, out_dir: Path) -> None:
    viz.apply_style()
    import matplotlib.pyplot as plt

    X, y = make_xy(primary)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.SEED
    )
    cv = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)

    # --- model comparison on training data only ---
    results: list[ModelResult] = []
    models = candidate_models()
    for name, pipe in models.items():
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        results.append(ModelResult(name, scores.mean(), scores.std()))
        print(f"  CV AUC {name}: {scores.mean():.3f} +/- {scores.std():.3f}")

    best = max(results, key=lambda r: r.cv_auc_mean)
    pipe = models[best.name]

    # --- feature-group ablation (training split only) ---
    print("  running feature-group ablation...")
    ablation_base, ablation = ablation_table(X_train, y_train, cv)

    # --- threshold from out-of-fold training probabilities ---
    oof = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    threshold = pick_threshold(y_train.to_numpy(), oof, config.RECALL_TARGET)
    curve = _threshold_curve(y_train.to_numpy(), oof)

    # --- final fit and one-shot test evaluation ---
    pipe.fit(X_train, y_train)
    proba_test = pipe.predict_proba(X_test)[:, 1]
    pred_test = (proba_test >= threshold).astype(int)

    auc = roc_auc_score(y_test, proba_test)
    kappa = cohen_kappa_score(y_test, pred_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, pred_test, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_test, pred_test)

    # --- figures ---
    fig, ax = viz.new_fig(5.6, 4.6)
    for (name, m), color in zip(models.items(), viz.SERIES):
        if name == best.name:
            fpr, tpr, _ = roc_curve(y_test, proba_test)
            label = f"{name.replace('_', ' ')} (test AUC {auc:.3f})"
        else:
            m.fit(X_train, y_train)
            other_proba = m.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, other_proba)
            label = f"{name.replace('_', ' ')} (test AUC {roc_auc_score(y_test, other_proba):.3f})"
        ax.plot(fpr, tpr, color=color, label=label)
    ax.plot([0, 1], [0, 1], color=viz.GRID, linestyle="--", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC — mid-semester at-risk model (held-out test)")
    ax.legend(loc="lower right")
    fig.savefig(out_dir / "fig_roc.png")
    plt.close(fig)

    fig, ax = viz.new_fig()
    ax.plot(curve.threshold, curve.precision, color=viz.BLUE)
    ax.plot(curve.threshold, curve.recall, color=viz.GREEN)
    ax.axvline(threshold, color=viz.TEXT_SECONDARY, linestyle=":", linewidth=1.5)
    ax.text(threshold + 0.01, 0.05, f"chosen threshold = {threshold:.2f}",
            color=viz.TEXT_SECONDARY, fontsize=9)
    ax.text(curve.threshold.iloc[2], curve.recall.iloc[2] - 0.06, "recall",
            color=viz.GREEN, fontweight="bold")
    ax.text(curve.threshold.iloc[2], curve.precision.iloc[2] - 0.06, "precision",
            color=viz.BLUE, fontweight="bold")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score (out-of-fold, training split)")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Threshold chosen for recall ≥ {config.RECALL_TARGET:.0%} before touching the test set")
    fig.savefig(out_dir / "fig_threshold.png")
    plt.close(fig)

    # --- permutation importance (always) + SHAP (best-effort) ---
    perm = permutation_importance(
        pipe, X_test, y_test, scoring="roc_auc",
        n_repeats=10, random_state=config.SEED,
    )
    imp = (
        pd.Series(perm.importances_mean, index=X_test.columns)
        .loc[config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES]
        .sort_values()
        .tail(10)
    )
    fig, ax = viz.new_fig(7.0, 4.6)
    ax.barh(imp.index, imp.values, color=viz.BLUE, height=0.62)
    ax.set_xlabel("Mean AUC drop when column is shuffled (test split)")
    ax.set_title("Permutation importance — what the model actually uses")
    ax.grid(axis="y", visible=False)
    fig.savefig(out_dir / "fig_permutation_importance.png")
    plt.close(fig)

    coef_table = _coefficient_table(pipe)

    # --- subgroup metrics for the shipped model ---
    groups = subgroup_table(X_test, y_test.to_numpy(), pred_test)

    # --- metrics report ---
    lines = [
        "# Model report — mid-semester at-risk prediction (primary file)",
        "",
        f"- Target: at-risk = Grade in {sorted(config.AT_RISK_GRADES)} "
        f"(base rate {y.mean():.1%})",
        f"- Split: stratified {1 - config.TEST_SIZE:.0%}/{config.TEST_SIZE:.0%}; "
        f"all selection on the training split only",
        f"- Model inputs: {len(config.NUMERIC_FEATURES) + len(config.CATEGORICAL_FEATURES)} "
        "mid-semester features; demographic/background columns excluded by design "
        "(see ablation below)",
        "",
        "## Model comparison (5-fold cross-validation on training split, ROC AUC)",
        "",
        *[f"- {r.name}: {r.cv_auc_mean:.3f} +/- {r.cv_auc_std:.3f}"
          + ("  <- selected" if r.name == best.name else "") for r in results],
        "",
        "## Feature-group ablation (5-fold cross-validation on training split, logistic regression)",
        "",
        f"Baseline with every candidate column, demographics included: "
        f"cross-validation (CV) AUC {ablation_base:.3f}. Each group is then evaluated alone and "
        f"removed (model retrained each time):",
        "",
        ablation.to_markdown(index=False),
        "",
        "Demographics & background sit at chance level alone and cost nothing to",
        "remove, so they are excluded from the model's inputs — an evaluated",
        "decision, not an omission. The subgroup check below still audits the",
        "model's outcomes by those attributes.",
        "",
        "## Held-out test performance (selected model)",
        "",
        f"- ROC AUC: {auc:.3f}",
        f"- Operating threshold: {threshold:.2f} "
        f"(chosen on training out-of-fold [OOF] probabilities for recall ≥ {config.RECALL_TARGET:.0%})",
        f"- Recall: {recall:.3f} | Precision: {precision:.3f} | "
        f"F1: {f1:.3f} | Cohen's kappa: {kappa:.3f}",
        "",
        "Confusion matrix (rows = actual, cols = predicted; positive = at-risk):",
        "",
        f"| | pred not-at-risk | pred at-risk |",
        f"|---|---|---|",
        f"| **actual not-at-risk** | {cm[0, 0]} | {cm[0, 1]} |",
        f"| **actual at-risk** | {cm[1, 0]} | {cm[1, 1]} |",
        "",
        "## Subgroup check (test split, shipped model at operating threshold)",
        "",
        groups.to_markdown(index=False),
        "",
        "See [WRITEUP.md](../WRITEUP.md) for interpretation of any "
        "recall/false positive rate (FPR)/flag-rate gaps, and",
        "audit_report.md for the biased file, where the failure mode is different.",
    ]
    if coef_table is not None:
        lines += [
            "",
            "## Largest standardized coefficients (selected logistic regression)",
            "",
            coef_table.to_markdown(),
            "",
            "Positive = pushes toward at-risk. Coefficients are on scaled inputs,",
            "so magnitudes are comparable across features.",
        ]
    lines += [
        "",
        "## Figures",
        "",
        "- [fig_roc.png](fig_roc.png) — ROC curves, both candidates, test split",
        "- [fig_threshold.png](fig_threshold.png) — precision/recall vs. threshold with the chosen operating point",
        "- [fig_permutation_importance.png](fig_permutation_importance.png) — importance on the original columns",
    ]

    out_path = out_dir / "model_report.md"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"  wrote {out_path}")
