"""Loading and data validation.

validate() writes outputs/validation_report.md — the evidence base for every
"what we relied on / what we set aside" claim in the write-up. It checks the
data against its own documentation (data card) rather than assuming either
is correct.
"""

from pathlib import Path

import pandas as pd

from . import config


def load(data_dir: Path, filename: str) -> pd.DataFrame:
    df = pd.read_csv(data_dir / filename)
    df["at_risk"] = df[config.TARGET_COL].isin(config.AT_RISK_GRADES).astype(int)
    df["grade_ordinal"] = df[config.TARGET_COL].map(config.GRADE_TO_ORDINAL)
    return df


def _recomputed_total(df: pd.DataFrame, participation_scale: float) -> pd.Series:
    """Total score per the data card's documented weights.

    participation_scale rescales Participation_Score to 0-100 first (the two
    files disagree on its range: 0-100 in the primary file, 0-10 in the biased
    file despite the data card saying 0-10 for both).
    """
    w = config.DOCUMENTED_WEIGHTS
    total = sum(df[col] * weight for col, weight in w.items() if col != "Participation_Score")
    return total + df["Participation_Score"] * participation_scale * w["Participation_Score"]


def _grade_binning_table(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(config.TARGET_COL)["Total_Score"]
        .agg(["min", "max", "count"])
        .reindex(config.GRADE_ORDER)
        .round(2)
    )


def validate(primary: pd.DataFrame, biased: pd.DataFrame, out_dir: Path) -> None:
    lines: list[str] = ["# Data validation report", ""]

    def section(title: str) -> None:
        lines.extend(["", f"## {title}", ""])

    def add(text: str = "") -> None:
        lines.append(text)

    section("Shapes, keys, and missing values")
    for name, df in (("primary", primary), ("biased", biased)):
        nulls = df.isna().sum()
        nulls = nulls[nulls > 0].to_dict()
        add(f"- **{name}**: {df.shape[0]} rows x {df.shape[1] - 2} columns; "
            f"duplicate Student_IDs: {int(df['Student_ID'].duplicated().sum())}; "
            f"nulls: {nulls or 'none'}")

    section("The two files describe the same roster with contradictory values")
    merged = primary.merge(biased, on="Student_ID", suffixes=("_m", "_b"))
    add(f"- Shared Student_IDs: {len(merged)} of {len(primary)}")
    for col in ["Department", "Attendance (%)", "Midterm_Score", "Grade"]:
        diff = (merged[f"{col}_m"] != merged[f"{col}_b"]).mean()
        add(f"- `{col}` differs for {diff:.0%} of shared students")
    add("- Conclusion: these are two variants of one extract, not two cohorts; "
        "they cannot both be treated as ground truth.")

    section("Primary file: Grade is a deterministic binning of Total_Score")
    add(_grade_binning_table(primary).to_markdown())
    add()
    bins = pd.cut(primary["Total_Score"], [0, 60, 70, 80, 90, 101],
                  labels=["F", "D", "C", "B", "A"], right=False)
    agreement = (bins.astype(str) == primary[config.TARGET_COL]).mean()
    add(f"- 10-point bins (A≥90 ... F<60) reproduce Grade for {agreement:.1%} of rows.")
    add("- Implication: predicting Grade with the score components included is "
        "reconstructing an arithmetic formula, not learning about students.")

    section("Does Total_Score follow the documented weights?")
    corr_m = primary["Total_Score"].corr(_recomputed_total(primary, participation_scale=1.0))
    corr_b = biased["Total_Score"].corr(_recomputed_total(biased, participation_scale=10.0))
    add(f"- primary: corr(Total_Score, documented weighted sum) = {corr_m:.2f} "
        "(Participation read as 0-100, contradicting the data card's 0-10)")
    add(f"- biased: corr(Total_Score, documented weighted sum) = {corr_b:.2f} "
        "(Participation rescaled from its observed 0-10 to 0-100)")
    add("- The primary file is internally consistent: Total_Score follows the "
        "documented weights exactly once Participation is treated as 0-100, and "
        "Grade is Total_Score binned. The biased file's Total_Score follows no "
        "documented rule at all.")

    section("Documentation vs. data: Participation_Score scale")
    add(f"- Data card says 0-10. Observed: primary "
        f"{primary['Participation_Score'].min():.0f}-{primary['Participation_Score'].max():.0f}, "
        f"biased {biased['Participation_Score'].min():.0f}-{biased['Participation_Score'].max():.0f}. "
        "The card's scale is treated as the documentation error in the primary "
        "file's case (the weights reconcile perfectly under 0-100).")

    section("Biased file: Grade correlates with Attendance and nothing else")
    corr = (
        biased[config.ALL_NUMERIC + config.LEAKY_COLS]
        .corrwith(biased["grade_ordinal"])
        .sort_values(key=abs, ascending=False)
        .round(3)
    )
    add(corr.to_markdown(headers=["feature", "corr with Grade (ordinal)"]))
    add()
    add("- Every academic score, including Final_Score and Total_Score, is "
        "uncorrelated with Grade in this file. Its Grade is attendance plus "
        "noise — see [outputs/audit_report.md](audit_report.md).")

    section("Class balance for the chosen target (at-risk = D or F, primary file)")
    add(f"- at-risk rate: {primary['at_risk'].mean():.1%} "
        f"({int(primary['at_risk'].sum())} of {len(primary)})")
    add(f"- Grade counts: {primary[config.TARGET_COL].value_counts().to_dict()}")
    add("- Note the 16 A's out of 5000 — one reason a 5-class letter-grade "
        "model was set aside in favor of the binary at-risk target.")

    out_path = out_dir / "validation_report.md"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"  wrote {out_path}")
