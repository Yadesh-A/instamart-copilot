# Instamart Customer Intelligence & Support Copilot

## Overview
This repository contains a 4-part data and AI pipeline that transforms raw e-commerce data into a clean analytical database, segments customers, predicts churn, triages support tickets using LLMs, and deploys an autonomous ReAct refund agent.

## Reproduce
1. `python -m venv venv`
2. Activate environment: `.\venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Mac/Linux)
3. `pip install -r requirements.txt`
4. `python generate_data.py`
5. `python part_a_cleaning.py`
*(Add Part B, C, D execution commands here as you build them)*

## Part A (Data Foundation)
* **Raw Audit:** 13,084 order rows; 119 exact duplicates; 656 pending rows... *(Add the rest of your printout numbers here)*
* **Cleaning Rules:**
  * Standardized city strings via `.strip().title()` (e.g., mapping " chennai " to "Chennai").
  * Mapped irregular gender spellings to standard categories or "unknown".
  * Filled age outliers with group-based median transforms...
*(Proceed to fill out the SQL answers and EDA correlation figures once you run the cross-check script).*

## Part B (Customer Segments and Lapse Prediction)

### 1. RFM & K-Means Clustering
* **RFM Summary:** Total 1,175 customers. Recency median 19 days. Frequency mean 9.46. Monetary mean INR 2,552.06.
* **Clustering Selection:** K=4 was chosen based on an elbow at K=4 (Inertia ~743.1) and a peak silhouette score of ~0.526. While the inertia of K=8 (366.5) is lower than K=2 (1820.8), inertia alone cannot be used to pick K because it will always decrease as clusters increase (eventually reaching 0 if every point is its own cluster).
* **Segment Profiles (K=4):**
  * **Loyal/Frequent (Size ~436):** High frequency, moderate spend. Action: Loyalty rewards.
  * **Premium (Size ~138):** Highest monetary spend (mean INR 7,129.69). Action: VIP white-glove support.
  * **At-Risk/Rare (Size ~419):** High recency (days since last order), very low frequency. Action: Win-back discount campaigns.
  * **Recent/Low-Spend (Size ~182):** Low recency, low spend. Action: Cross-sell to increase basket size.
* **PCA:** The 2-component PCA explained variance ratios are approximately 0.634 and 0.218 (Total: ~0.852).

### 2. Lapse Prediction Preprocessing
* **Cohort:** 1,156 customers; 443 lapsed (38.32%). Baseline accuracy is 61.6% with a lapsed-class F1 of 0.
* **Scaler Difference:** The fitted scaler frequency mean (8.1158) differs slightly from the full cohort mean (8.0285) because the scaler was strictly fitted only on the 80% training data to prevent data leakage from the test set. 
* **Strict Independence:** No encoder, scaler, or resampler saw the test rows. No data from 2025-09-01 onwards entered the features. Segment labels from Task 3 were excluded from features to prevent leakage.

### 3. Model Tuning & Evaluation
* **Logistic Regression:** Best parameters `{'C': 0.1}`, CV F1: 0.639.
* **Random Forest:** Best parameters `{'max_depth': 5, 'min_samples_leaf': 10}`, CV F1: 0.670.
* **Model Choice:** Random Forest was selected (Test F1: ~0.706). Missing a lapsing customer (False Negative) means losing future revenue, while wrongly targeting a loyal one (False Positive) just costs a small marketing discount. Random Forest provides a strong balance of precision and recall for this business case.
* **Feature Importance (Top 5):**
  1. `recency`: Higher days since last order strongly signals churn.
  2. `frequency`: Fewer total orders indicates a lack of habit formation.
  3. `monetary`: Low overall spend correlates with low commitment.
  4. `tenure`: Newer customers are more likely to lapse than long-term ones.
  5. `avg_discount`: Customers relying heavily on discounts may leave when promotions end.

### 4. Class-Balance Experiment
Balancing the classes using oversampling caused **Recall to move the most**, jumping from ~0.809 to ~0.944, while Precision dropped. For the retention team, this means the balanced model is much better at catching almost all lapsing customers, even though it will accidentally send win-back discounts to a few loyal customers.

## Part C (Support Ticket Triage with Guardrails)

### 1. Input Guardrails & PII Redaction
* **Guardrails:** 4 empty tickets were skipped. 4 excessively long tickets (estimating up to ~734 tokens) were flagged and blocked from LLM processing to prevent context overflow.
* **Privacy (PII):** We successfully redacted 55 emails, 91 phone numbers, and 132 order IDs using Regex substitutions `[EMAIL]`, `[PHONE]`, and `[ORDER_ID]`. 0 PII elements leaked into the LLM prompts.

### 2. Resilience and Output Validation
* **Retry Logic:** The pipeline implements exponential backoff with jitter (Base Delay = 0.05s). During a full run, it successfully caught and recovered from exactly 19 `TransientLLMError` API timeouts. 
* **Validation:** Out of the 156 processed tickets:
  * 135 returned valid JSON (`classified_json`).
  * 12 returned chatty text where the category was safely extracted via keyword matching (`classified_keyword`).
  * 9 returned hallucinated or invalid categories and were safely flagged for human review (`flagged_invalid_output`).

### 3. Prompt Engineering: Zero-Shot vs Few-Shot
* **Zero-Shot:** Reached 84.6% accuracy, costing ~65.8 tokens per classified ticket. It struggled severely with edge cases, misclassifying 8 Fraud and 6 Technical tickets as Billing.
* **Few-Shot:** Reached 100.0% accuracy, costing ~128.7 tokens per classified ticket.
* **Business Justification:** The Few-Shot prompt costs roughly 63 extra tokens per ticket. This cost is absolutely worth paying because misrouting a `fraud` ticket to the `billing` queue delays critical security responses, risking massive financial loss and compliance violations.

### 4. Budget & Routing
* **Cost Control:** Running the Zero-Shot pipeline with a strict budget of 6,000 tokens successfully demonstrated the cutoff, halting exactly 60 tickets with a `budget_exhausted` status.
* **Routing Logic:** Outputs were routed dynamically. Invalid or exhausted tickets went to `human_review`. Fraud tickets were escalated to `security_team` (HIGH priority). Premium customers (based on Part B segments) were routed to `premium_desk` (HIGH priority), and all others went to the standard `support_queue`.

## Part D (RAG & Agent)
*(To be completed)*

## Limitations and Next Steps
* The Premium flag is a post-hoc whole-window label, making the retention list illustrative rather than a strict leakage-free evaluation.