"""Central configuration: file names, feature roles, and modeling constants.

Feature roles implement the mid-semester snapshot scenario (see WRITEUP.md):
the model may only use information an advisor would plausibly have shortly
after midterms. Final_Score, Projects_Score, and Total_Score are end-of-course
quantities and Grade is computed from them, so they are excluded as leakage.
"""

from pathlib import Path

PRIMARY_CSV = "Students Performance Dataset.csv"
BIASED_CSV = "Students_Grading_Dataset_Biased.csv"

ID_COLS = ["Student_ID", "First_Name", "Last_Name", "Email"]

# Model inputs. Demographic/background columns are excluded by design: the
# feature-group ablation (model_report.md) shows they carry no signal alone
# and their removal costs nothing, so the shipped model never sees them.
NUMERIC_FEATURES = [
    "Midterm_Score",
    "Quizzes_Avg",
    "Assignments_Avg",
    "Participation_Score",
    "Attendance (%)",
    "Study_Hours_per_Week",
    "Sleep_Hours_per_Night",
    "Stress_Level (1-10)",
]

CATEGORICAL_FEATURES = [
    "Department",
    "Extracurricular_Activities",
]

# Demographic/background columns: evaluated in the ablation and used by the
# subgroup audit, but never model inputs.
DEMOGRAPHIC_NUMERIC = ["Age"]
DEMOGRAPHIC_CATEGORICAL = [
    "Gender",
    "Internet_Access_at_Home",
    "Parent_Education_Level",
    "Family_Income_Level",
]

# Full numeric roster for data-validation correlation tables (the data itself
# doesn't change just because the model's inputs do).
ALL_NUMERIC = NUMERIC_FEATURES + DEMOGRAPHIC_NUMERIC

# Feature-group ablation over every candidate column (model inputs +
# demographics). Groups follow the write-up's narrative: the four graded
# components together, attendance on its own.
FEATURE_GROUPS = {
    "Graded components (midterm, quizzes, assignments, participation)": [
        "Midterm_Score", "Quizzes_Avg", "Assignments_Avg", "Participation_Score",
    ],
    "Attendance": ["Attendance (%)"],
    "Self-reported habits (study, sleep, stress)": [
        "Study_Hours_per_Week", "Sleep_Hours_per_Night", "Stress_Level (1-10)",
    ],
    "Context (department, extracurriculars)": [
        "Department", "Extracurricular_Activities",
    ],
    "Demographics & background (age, gender, income, parent ed., internet)":
        DEMOGRAPHIC_NUMERIC + DEMOGRAPHIC_CATEGORICAL,
}

LEAKY_COLS = ["Final_Score", "Projects_Score", "Total_Score"]

TARGET_COL = "Grade"
AT_RISK_GRADES = {"D", "F"}

SENSITIVE_ATTRS = ["Gender", "Family_Income_Level", "Internet_Access_at_Home"]

SEED = 42
TEST_SIZE = 0.20
N_FOLDS = 5
RECALL_TARGET = 0.80  # advisor-oriented: missing an at-risk student costs more
                      # than a false alert, so tune the threshold for recall

GRADE_ORDER = ["F", "D", "C", "B", "A"]
GRADE_TO_ORDINAL = {"F": 0, "D": 1, "C": 2, "B": 3, "A": 4}

# Documented component weights from the dataset's data card (used only to test
# whether Total_Score actually follows them — it does not, in either file).
DOCUMENTED_WEIGHTS = {
    "Midterm_Score": 0.15,
    "Final_Score": 0.25,
    "Assignments_Avg": 0.15,
    "Quizzes_Avg": 0.10,
    "Participation_Score": 0.05,
    "Projects_Score": 0.30,
}


def resolve_data_dir(cli_value: str | None) -> Path:
    """Locate the dataset. Order: explicit --data-dir, ./data, ../data."""
    if cli_value:
        p = Path(cli_value)
        if not (p / PRIMARY_CSV).exists():
            raise FileNotFoundError(f"{PRIMARY_CSV!r} not found in {p.resolve()}")
        return p
    for candidate in (Path("data"), Path("..") / "data"):
        if (candidate / PRIMARY_CSV).exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find the dataset. Put {PRIMARY_CSV!r} (from the Kaggle "
        "'students-grading-dataset' package) in the data/ folder, or pass "
        "--data-dir pointing at the folder that contains it."
    )
