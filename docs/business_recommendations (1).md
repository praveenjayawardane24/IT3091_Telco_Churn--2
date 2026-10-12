# High-Risk Customer Profiling and Retention Recommendations

**Confirmed final run:** Gradient Boosting, 79 actual churners among the 100 highest-ranked held-out customers; precision@100 = 0.79 and lift@100 = 2.97623.

## Retrospective top-100 risk profile
- Average tenure: **3.33 months** vs **31.92** across test customers.
- Average monthly charges: **80.98** vs **64.09** across test customers.
- Month-to-month contract: **100%** of top 100 vs **54.86%** of test customers.
- Fiber-optic internet: **93%** vs **43.51%**.
- No online security: **99%** vs **49.75%**.
- No technical support: **98%** vs **49.82%**.
- Electronic-check payment: **85%** vs **33.64%**.

## Model importance (individual encoded predictors)
`Contract_Month-to-month` (0.4179), `tenure` (0.1296), `InternetService_Fiber optic` (0.0954), `OnlineSecurity_No` (0.0794), `TechSupport_No` (0.0630), `MonthlyCharges` (0.0508), `PaymentMethod_Electronic check` (0.0491).

## Recommendations to test
1. Early onboarding and proactive check-ins for newer customers.
2. Optional incentives for longer contracts, with transparent terms.
3. Investigate high-bill concerns and test targeted retention offers.
4. Offer relevant technical-support and online-security assistance.
5. Investigate payment friction and make convenient alternatives available.
6. Prioritize a capacity-limited outreach shortlist, then evaluate campaign results with controlled experiments.

## Responsible AI and limitations
- Observed associations and impurity-based feature importance are not causal explanations.
- The top-100 shortlist is drawn from historical held-out test records, not a live customer list.
- Predicted churn probabilities have not been checked for calibration; use primarily for ranking until validated.
- Review subgroup performance and fairness before operational use, especially where demographic variables are present.
- Retention offers should not discriminate unfairly or use sensitive information without appropriate safeguards.
- Check model drift and retrain with more recent data before deployment.
