# Data validation report


## Shapes, keys, and missing values

- **primary**: 5000 rows x 23 columns; duplicate Student_IDs: 0; nulls: {'Parent_Education_Level': 1025}
- **biased**: 5000 rows x 23 columns; duplicate Student_IDs: 0; nulls: {'Attendance (%)': 516, 'Assignments_Avg': 517, 'Parent_Education_Level': 1794}

## The two files describe the same roster with contradictory values

- Shared Student_IDs: 5000 of 5000
- `Department` differs for 74% of shared students
- `Attendance (%)` differs for 100% of shared students
- `Midterm_Score` differs for 100% of shared students
- `Grade` differs for 82% of shared students
- Conclusion: these are two variants of one extract, not two cohorts; they cannot both be treated as ground truth.

## Primary file: Grade is a deterministic binning of Total_Score

| Grade   |   min |   max |   count |
|:--------|------:|------:|--------:|
| F       | 50.6  | 59.99 |     279 |
| D       | 60.01 | 70    |    1760 |
| C       | 70    | 80    |    2307 |
| B       | 80    | 89.97 |     638 |
| A       | 90.09 | 95.09 |      16 |

- 10-point bins (A≥90 ... F<60) reproduce Grade for 100.0% of rows.
- Implication: predicting Grade with the score components included is reconstructing an arithmetic formula, not learning about students.

## Does Total_Score follow the documented weights?

- primary: corr(Total_Score, documented weighted sum) = 1.00 (Participation read as 0-100, contradicting the data card's 0-10)
- biased: corr(Total_Score, documented weighted sum) = -0.01 (Participation rescaled from its observed 0-10 to 0-100)
- The primary file is internally consistent: Total_Score follows the documented weights exactly once Participation is treated as 0-100, and Grade is Total_Score binned. The biased file's Total_Score follows no documented rule at all.

## Documentation vs. data: Participation_Score scale

- Data card says 0-10. Observed: primary 0-100, biased 0-10. The card's scale is treated as the documentation error in the primary file's case (the weights reconcile perfectly under 0-100).

## Biased file: Grade correlates with Attendance and nothing else

| feature               |   corr with Grade (ordinal) |
|:----------------------|----------------------------:|
| Attendance (%)        |                       0.607 |
| Quizzes_Avg           |                      -0.03  |
| Final_Score           |                      -0.028 |
| Assignments_Avg       |                      -0.028 |
| Stress_Level (1-10)   |                       0.025 |
| Total_Score           |                      -0.024 |
| Age                   |                      -0.019 |
| Sleep_Hours_per_Night |                      -0.018 |
| Participation_Score   |                      -0.017 |
| Midterm_Score         |                      -0.014 |
| Projects_Score        |                      -0.014 |
| Study_Hours_per_Week  |                       0.003 |

- Every academic score, including Final_Score and Total_Score, is uncorrelated with Grade in this file. Its Grade is attendance plus noise — see [outputs/audit_report.md](audit_report.md).

## Class balance for the chosen target (at-risk = D or F, primary file)

- at-risk rate: 40.8% (2039 of 5000)
- Grade counts: {'C': 2307, 'D': 1760, 'B': 638, 'F': 279, 'A': 16}
- Note the 16 A's out of 5000 — one reason a 5-class letter-grade model was set aside in favor of the binary at-risk target.
