"""Bias / data-quality audit of the biased file.

The point is not to model this file well — it is to show *why it must not be
modeled*: its Grade is driven by Attendance and essentially nothing else, so
any model trained on it operationalizes an attendance bias while wearing the
costume of an academic-outcome predictor.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from . import config, viz
from .features import make_xy
from .model import candidate_models


def run(biased: pd.DataFrame, out_dir: Path) -> None:
    viz.apply_style()
    import matplotlib.pyplot as plt

    lines = ["# Audit report — the biased file", ""]

    # --- 1. what actually correlates with Grade ---
    score_cols = config.ALL_NUMERIC + config.LEAKY_COLS
    corr = (
        biased[score_cols]
        .corrwith(biased["grade_ordinal"])
        .sort_values(key=abs, ascending=False)
        .round(3)
    )
    lines += [
        "## Grade correlates with Attendance and nothing else",
        "",
        corr.to_markdown(headers=["feature", "corr with Grade (ordinal)"]),
        "",
        "Every academic score — including Final_Score and the Total_Score that "
        "Grade is supposedly computed from — sits at |r| < 0.05. The data card "
        "admits bias was injected ('students with high attendance get slightly "
        "better grades'); in fact, attendance is the *only* signal.",
        "",
    ]

    fig, ax = viz.new_fig(7.0, 4.2)
    c = corr.sort_values()
    colors = [viz.BLUE if abs(v) > 0.1 else viz.GRID for v in c.values]
    ax.barh(c.index, c.values, color=colors, height=0.62)
    ax.axvline(0, color=viz.TEXT_SECONDARY, linewidth=1)
    ax.set_xlabel("Correlation with Grade (ordinal)")
    ax.set_title("Biased file: only Attendance predicts Grade")
    ax.grid(axis="y", visible=False)
    fig.savefig(out_dir / "fig_audit_correlations.png")
    plt.close(fig)

    # --- 2. attendance distribution by grade ---
    order = config.GRADE_ORDER
    fig, ax = viz.new_fig(6.4, 4.0)
    data = [biased.loc[biased["Grade"] == g, "Attendance (%)"].dropna() for g in order]
    parts = ax.boxplot(data, tick_labels=order, patch_artist=True, widths=0.5,
                       medianprops={"color": viz.TEXT_PRIMARY, "linewidth": 1.5})
    for box in parts["boxes"]:
        box.set(facecolor=viz.BLUE, alpha=0.55, edgecolor=viz.BLUE)
    ax.set_xlabel("Grade")
    ax.set_ylabel("Attendance (%)")
    ax.set_title("Biased file: the grade 'ladder' is an attendance ladder")
    fig.savefig(out_dir / "fig_audit_attendance_by_grade.png")
    plt.close(fig)

    med = biased.groupby("Grade")["Attendance (%)"].median().reindex(order).round(1)
    lines += [
        "## Attendance by grade",
        "",
        med.to_markdown(headers=["Grade", "median Attendance (%)"]),
        "",
    ]

    # --- 3. what a naive team would ship ---
    X, y = make_xy(biased)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.SEED
    )
    pipe = candidate_models()["gradient_boosting"]
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    auc_full = roc_auc_score(y_test, proba)

    rng = np.random.default_rng(config.SEED)
    X_train_na = X_train.copy()
    X_test_na = X_test.copy()
    X_train_na["Attendance (%)"] = rng.permutation(X_train_na["Attendance (%)"].to_numpy())
    X_test_na["Attendance (%)"] = rng.permutation(X_test_na["Attendance (%)"].to_numpy())
    pipe_no_att = candidate_models()["gradient_boosting"]
    pipe_no_att.fit(X_train_na, y_train)
    auc_no_att = roc_auc_score(y_test, pipe_no_att.predict_proba(X_test_na)[:, 1])

    lines += [
        "## What a model trained on this file actually learns",
        "",
        f"- Gradient boosting, same mid-semester pipeline as the main model: "
        f"test AUC **{auc_full:.3f}**",
        f"- Same model with Attendance values row-shuffled (column kept, signal "
        f"destroyed): test AUC **{auc_no_att:.3f}** (chance = 0.5)",
        "",
        "The model's apparent skill is entirely the injected attendance artifact. "
        "Neutralize that one column, and it collapses to a coin flip — there is no "
        "academic signal left to find.",
        "",
    ]

    # --- 4. does the artifact load onto sensitive groups? ---
    rows = []
    for attr in config.SENSITIVE_ATTRS:
        for value, g in biased.groupby(attr, dropna=False):
            rows.append({
                "attribute": attr,
                "group": str(value),
                "n": len(g),
                "mean_attendance": g["Attendance (%)"].mean(),
                "at_risk_rate": g["at_risk"].mean(),
            })
    grp = pd.DataFrame(rows).round(3)
    max_gap = (
        grp.groupby("attribute")["mean_attendance"]
        .agg(lambda s: s.max() - s.min())
        .max()
    )
    if max_gap < 3.0:
        exposure_verdict = (
            f"Mean attendance differs by at most {max_gap:.1f} points across "
            "groups of any audited attribute, so the injected bias does not "
            "concentrate on a protected group in this extract. That is luck, "
            "not safety: the failure is an integrity failure (grades decoupled "
            "from academic work), and the same mechanism would become a "
            "fairness failure the moment attendance correlates with, say, "
            "income or home internet access — as it often does in real cohorts."
        )
    else:
        exposure_verdict = (
            f"Mean attendance differs by up to {max_gap:.1f} points across "
            "groups, so the injected attendance artifact also loads unevenly "
            "onto sensitive groups — the integrity failure doubles as a "
            "fairness failure in this extract. See the table above for which "
            "groups are most exposed."
        )
    lines += [
        "## Exposure of sensitive groups to the artifact",
        "",
        grp.to_markdown(index=False),
        "",
        exposure_verdict,
        "",
        "## Verdict",
        "",
        "Set this file aside for outcome modeling. Use it as a worked example "
        "of why label provenance checks belong before any training run.",
    ]

    out_path = out_dir / "audit_report.md"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"  wrote {out_path}")
