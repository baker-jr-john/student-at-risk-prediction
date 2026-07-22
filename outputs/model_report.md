# Model report — mid-semester at-risk prediction (masked file)

- Target: at-risk = Grade in ['D', 'F'] (base rate 40.8%)
- Split: stratified 80%/20%; all selection on the training split only
- Model inputs: 10 mid-semester features; demographic/background columns excluded by design (see ablation below)

## Model comparison (5-fold CV on training split, ROC AUC)

- logistic_regression: 0.748 +/- 0.010  <- selected
- gradient_boosting: 0.714 +/- 0.006

## Feature-group ablation (5-fold CV on training split, logistic regression)

Baseline with every candidate column, demographics included: CV AUC 0.747. Each group is then evaluated alone and removed (model retrained each time):

| group                                                                 |   CV AUC (group alone) |   CV AUC (group removed) |   AUC cost of removal |
|:----------------------------------------------------------------------|-----------------------:|-------------------------:|----------------------:|
| Graded components (midterm, quizzes, assignments, participation)      |                  0.75  |                    0.487 |                 0.26  |
| Attendance                                                            |                  0.51  |                    0.747 |                 0     |
| Self-reported habits (study, sleep, stress)                           |                  0.492 |                    0.747 |                -0     |
| Context (department, extracurriculars)                                |                  0.503 |                    0.748 |                -0.001 |
| Demographics & background (age, gender, income, parent ed., internet) |                  0.478 |                    0.748 |                -0.001 |

Demographics & background sit at chance-level alone and cost nothing to
remove, so they are excluded from the model's inputs — an evaluated
decision, not an omission. The subgroup check below still audits the
model's outcomes by those attributes.

## Held-out test performance (selected model)

- ROC AUC: **0.757**
- Operating threshold: 0.33 (chosen on training OOF probabilities for recall >= 80%)
- Recall: **0.819** | Precision: **0.563** | F1: 0.667 | Cohen's kappa: 0.356

Confusion matrix (rows = actual, cols = predicted; positive = at-risk):

| | pred not-at-risk | pred at-risk |
|---|---|---|
| **actual not-at-risk** | 333 | 259 |
| **actual at-risk** | 74 | 334 |

## Subgroup check (test split, shipped model at operating threshold)

| attribute               | group   |   n |   base_rate |   flag_rate |   recall_tpr |   fpr |
|:------------------------|:--------|----:|------------:|------------:|-------------:|------:|
| Gender                  | Female  | 492 |       0.388 |       0.567 |        0.801 | 0.419 |
| Gender                  | Male    | 508 |       0.427 |       0.618 |        0.834 | 0.457 |
| Family_Income_Level     | High    | 339 |       0.404 |       0.631 |        0.839 | 0.49  |
| Family_Income_Level     | Low     | 331 |       0.405 |       0.55  |        0.791 | 0.386 |
| Family_Income_Level     | Medium  | 330 |       0.415 |       0.597 |        0.825 | 0.435 |
| Internet_Access_at_Home | No      | 495 |       0.436 |       0.592 |        0.815 | 0.419 |
| Internet_Access_at_Home | Yes     | 505 |       0.38  |       0.594 |        0.823 | 0.454 |

See WRITEUP.md for interpretation of any recall/FPR/flag-rate gaps, and
audit_report.md for the biased file, where the failure mode is different.

## Largest standardized coefficients (selected logistic regression)

|                                |   coef |
|:-------------------------------|-------:|
| Assignments_Avg                | -0.665 |
| Midterm_Score                  | -0.643 |
| Quizzes_Avg                    | -0.361 |
| Participation_Score            | -0.351 |
| Extracurricular_Activities_No  | -0.157 |
| Department_Business            | -0.15  |
| Extracurricular_Activities_Yes | -0.101 |
| Department_CS                  | -0.053 |
| Department_Engineering         | -0.036 |
| Attendance (%)                 |  0.036 |
| Study_Hours_per_Week           | -0.031 |
| Department_Mathematics         | -0.018 |

Positive = pushes toward at-risk. Coefficients are on scaled inputs,
so magnitudes are comparable across features.

## Figures

- fig_roc.png — ROC curves, both candidates, test split
- fig_threshold.png — precision/recall vs. threshold with the chosen operating point
- fig_permutation_importance.png — importance on the original columns
