# Audit report — the biased file

## Grade correlates with Attendance and nothing else

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

Every academic score — including Final_Score and the Total_Score that Grade is supposedly computed from — sits at |r| < 0.05. The data card admits bias was injected ('students with high attendance get slightly better grades'); in fact attendance is the *only* signal.

## Attendance by grade

| Grade   |   median Attendance (%) |
|:--------|------------------------:|
| F       |                    66.1 |
| D       |                    61.8 |
| C       |                    67.1 |
| B       |                    83.6 |
| A       |                    88.7 |

## What a model trained on this file actually learns

- Gradient boosting, same mid-semester pipeline as the main model: test AUC **0.746**
- Same model with Attendance values row-shuffled (column kept, signal destroyed): test AUC **0.496** (chance = 0.5)

The model's apparent skill is entirely the injected attendance artifact. Neutralize that one column and it collapses to a coin flip — there is no academic signal left to find.

## Exposure of sensitive groups to the artifact

| attribute               | group   |    n |   mean_attendance |   at_risk_rate |
|:------------------------|:--------|-----:|------------------:|---------------:|
| Gender                  | Female  | 2449 |            75.527 |          0.343 |
| Gender                  | Male    | 2551 |            75.34  |          0.35  |
| Family_Income_Level     | High    | 1044 |            75.473 |          0.334 |
| Family_Income_Level     | Low     | 1983 |            75.137 |          0.354 |
| Family_Income_Level     | Medium  | 1973 |            75.705 |          0.346 |
| Internet_Access_at_Home | No      |  515 |            75.025 |          0.336 |
| Internet_Access_at_Home | Yes     | 4485 |            75.479 |          0.348 |

Mean attendance differs by at most 0.6 points across groups of any audited attribute, so the injected bias does not concentrate on a protected group in this extract. That is luck, not safety: the failure is an integrity failure (grades decoupled from academic work), and the same mechanism would become a fairness failure the moment attendance correlates with, say, income or home internet access — as it often does in real cohorts.

## Verdict

Set this file aside for outcome modeling. Use it as a worked example of why label provenance checks belong before any training run.
