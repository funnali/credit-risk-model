# Credit Risk Probability Model for Alternative Data

An end-to-end credit scoring system built for Bati Bank's buy-now-pay-later service,
using transaction data from the Xente eCommerce platform.

## Project Structure
credit-risk-model/
├── .github/workflows/ci.yml
├── data/raw/ and data/processed/
├── notebooks/eda.ipynb
├── src/data_processing.py, train.py, predict.py
├── src/api/main.py, pydantic_models.py
├── tests/test_data_processing.py
├── Dockerfile, docker-compose.yml
└── README.md

## Credit Scoring Business Understanding

### 1. How does the Basel II Accord influence the need for interpretable and well-documented models?

The Basel II Capital Accord requires banks to hold capital reserves proportional to
their credit risk exposure. To satisfy regulators, a bank must be able to explain
exactly how its model arrives at a risk score for any given customer. This means
black-box models are problematic in isolation: if a regulator or auditor asks why a
customer was denied credit, the bank must provide a clear, traceable answer.

Basel II's Internal Ratings-Based (IRB) approach specifically demands that models
be validated, documented, and monitored on an ongoing basis. Every modeling choice
— feature selection, transformation, algorithm — must be justified and recorded.
This directly drives the use of techniques like Weight of Evidence (WoE) and
Information Value (IV), which produce interpretable scorecards where the contribution
of each variable to the final score is transparent and auditable.

### 2. Why is a proxy variable necessary, and what business risks does it introduce?

The raw Xente transaction dataset contains no direct label indicating whether a
customer defaulted on a loan. Without a ground-truth default label, supervised
classification is impossible. A proxy variable bridges this gap by using observable
behavioral signals — Recency, Frequency, and Monetary (RFM) patterns — to
approximate creditworthiness. Customers who transact rarely and spend little are
treated as disengaged, and disengagement is used as a proxy for higher default risk.

However, this introduces serious business risks. First, the proxy may not actually
correlate with true default behavior — a low-frequency customer might simply be
a careful spender, not a bad borrower. Second, the model encodes the assumptions
baked into the clustering, meaning errors in the proxy definition propagate directly
into lending decisions. Third, regulatory scrutiny may increase because the target
variable itself is an engineering artifact, not an observed outcome. These limitations
must be disclosed explicitly in model documentation and monitored continuously.

### 3. What are the key trade-offs between interpretable and high-performance models?

| Dimension | Logistic Regression + WoE | Gradient Boosting (XGBoost) |
|---|---|---|
| Interpretability | High - each feature has a clear coefficient | Low - complex interactions are opaque |
| Regulatory acceptance | Strong - aligns with Basel II scorecard tradition | Weaker - needs extra explainability tools |
| Predictive performance | Moderate | Typically higher AUC |
| Auditability | Easy to document and validate | Requires SHAP or LIME for explanation |
| Overfitting risk | Low | Higher without careful tuning |
| Deployment simplicity | Very simple | More complex |

In a regulated banking context, the preferred approach is to use Logistic Regression
with WoE as the primary production model for its auditability, while using Gradient
Boosting as a benchmark to quantify the performance cost of interpretability.

## Setup

Install dependencies:
    pip install -r requirements.txt

Run the API:
    uvicorn src.api.main:app --reload