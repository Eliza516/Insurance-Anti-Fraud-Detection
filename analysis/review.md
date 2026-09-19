# Independent review: Insurance Anti-Fraud Detection

**Reviewed:** `analysis.ipynb`, `report.html`.
**Data:** `insurance_claims.csv` @ commit `7d1b322792ba45878c0f1a879e6aee8328939d34`.

## Checklist

| Item | Result | Notes |
|---|---|---|
| Reproducibility | ✅ Pass | Notebook runs end to end with no errors (`nbconvert --execute`). `SEED = 42` fixes the split, CV, models and bootstrap. Commit hash is recorded at the top of the notebook and the report. |
| Data split | ✅ Pass | 80/20 stratified split (fraud rate 24.8% / 24.5%). `assert` confirms no `policy_number` overlap between the two sets. A random split is appropriate — no repeated groups, and the data spans only 2 months. Test is used once, for the final model; the threshold is picked from OOF predictions on train. |
| Leakage | ✅ Pass | Scaler, one-hot and SelectKBest all sit inside the `Pipeline`, fit per fold. The transform outside the pipeline (`clean`) is deterministic — nothing learned from data. No post-outcome or label-derived features. Label-aware EDA runs on train only. |
| Drift | ✅ Pass | KS test on 8 numeric features: all p > 0.1. No separate unlabeled Kaggle test set, so adversarial validation isn't needed. |
| Data quality | ⚠️ Flag | `?` is kept as an explicit "Unknown" category, which is reasonable since missingness can itself carry information. "None" in `authorities_contacted` is correctly read as a real category, not NaN — pandas ≥2's default `read_csv` would silently turn 91 of these into missing values. Biggest open question: **the data looks simulated** — chess/cross-fit hobbies show ~86% fraud, which is hard to justify on business grounds. |
| Method | ✅ Pass | Includes a Dummy baseline. Primary metric is PR-AUC, appropriate for imbalanced data; F1, ROC-AUC, precision and recall are also reported. CV uses StratifiedKFold, matching the split strategy. `class_weight='balanced'` is used for LR, DT and RF. All 5 required models are covered, plus feature selection and a tree plot. |
| Overfitting | ⚠️ Flag | Random Forest shows train PR-AUC 1.0 vs. CV 0.65 — clear overfit; it wasn't selected. The final model (NB) has train 0.726 ≈ CV 0.726. **Test PR-AUC (0.551) sits 0.18 below CV (0.726).** Investigated via bootstrap CI on test ([0.43, 0.70]), repeated 5x5 CV (0.72 ± 0.05, min 0.63), and F1 agreement (test 0.73 vs. OOF 0.755). Conclusion: most of the gap comes from test-set variance (only 49 positives) and from ranking noise inside the flagged group, not leakage — though the test score still sits low in the CV distribution, so PR-AUC should be read with that uncertainty in mind. |
| Statistics | ✅ Pass | Chi-square tests report Cramér's V; all 4 tests hold up after a Bonferroni correction. Mann-Whitney on `total_claim_amount` has a small p-value but only ~9% median difference, and the report calls this out as a small effect. The 8 Mann-Whitney tests aren't multiple-comparison corrected; `umbrella_limit` (p = 0.038) wouldn't survive Bonferroni, but it isn't used in any conclusion. |
| Interpretation | ✅ Pass | The report is explicit that these are correlations, not causal claims, and that the model's predictions match a 2-condition rule 100% of the time rather than overselling model "intelligence." The top-5 list is a 4-method consensus, with a caveat that ranks 3-5 have small, unstable effects. |
| Numbers | ✅ Pass | Every figure in the report was checked against notebook output: CV table (0.726/0.712/0.707/0.646/0.628/0.262), test table (0.551/0.837/0.730/0.636/0.857/0.845), CIs (0.43-0.70; 0.63-0.81), group rates (60%, 10%, 15%, 8%, 87%, 86%, 4%, 88%), medians (60,995 / 56,080), Cramér's V (0.50 / 0.46), confusion matrix (42/49 caught, 66 flagged). |
| Submission format | — N/A | Not a competition-style task; no unlabeled Kaggle test set. |
| Visualization & communication | ✅ Pass | Charts have titles and axis labels. The tree uses raw units (USD) since scaling is skipped for tree models. The report notes that tree `value` counts are class-weighted. Recommendations are concrete and prioritized. |
| Data source | ✅ Pass | Matches the README's description (Kaggle, buntyshah). No secrets; no LFS pointers. The raw data has zip codes and incident addresses (simulated PII) — both are excluded from the model and don't appear in the report. |

## Fixes made during review

1. **Top-5 features.** The first pass ranked purely by permutation importance on the final model. Ranks 3-5 there had importance smaller than their own standard deviation (e.g. `authorities_contacted` 0.011 ± 0.013), too noisy to trust. Switched to a **4-method consensus ranking**, with a caveat added in both the notebook and the report.
2. **CV-test gap.** Added bootstrap CI, repeated CV, and a comparison against the 2-condition rule to explain the gap, instead of just reporting the numbers.
3. **Hard-to-read tree.** The first version showed thresholds on the scaled axis (e.g. `vehicle_claim <= -0.02`). Scaling is now skipped for DT and RF (tree models don't need it), so thresholds read in USD. CV score is unchanged.

## Risks & limitations

- **The "all models on test" table (section 8.1)** is reference-only, included because the assignment asks for a 5-model comparison. Using it to pick a model would break the "touch test once" rule — on that table KNN has the highest test PR-AUC (0.61), a different order than CV, which is itself a good illustration of how noisy a small test set is.
- **NB, DT and LR differences are within noise.** NB is selected on highest CV PR-AUC, but there's no clear statistical edge. If interpretability matters more, DT or the 2-condition rule are equally good picks.
- **The F1-optimal threshold** doesn't reflect the real-world cost tradeoff between false alarms and missed fraud.
- **Generalization:** 1,000 rows, 2 months, likely simulated. Shouldn't go straight to production.
- **Ethics & legal:** flagging fraud based on personal hobbies raises discrimination and privacy concerns.

## Suggested improvements

- **Data:** per-customer claim history, repair-shop and witness details, time from incident to report, and multi-year data for a time-based check.
- **Model:** probability calibration (NB tends to give skewed probabilities); cost-matrix-based threshold selection; a separate ranking model for the Major Damage group to improve precision there.
- **Evaluation:** nested or repeated CV instead of a single split; fairness checks across gender, age and occupation.

## Overall confidence: **Medium**

The pipeline itself is sound (no leakage, correct split, reproducible), and the qualitative conclusion (two dominant factors) is solid. The performance numbers carry wide uncertainty from the small sample, and real-world applicability is limited by the likely-simulated data.
