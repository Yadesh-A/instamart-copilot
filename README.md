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

## Part D (Policy RAG Assistant and Refund Agent)

### 1. Document Chunking & RAG Evaluation
* **Chunking:** The 5 policy text files were split into exactly 15 chunks (40 words each, 10-word overlap). 
* **Evaluation (Overlap 10):** 
  * Answerable Questions: 12/12 retrieved expected doc, 12/12 contained expected fact. 
  * Unanswerable Questions: 2/2 correctly triggered the "I could not find" fallback.
* **Evaluation (Overlap 0):** Zero overlap caused failures. For example, "What delivery time makes a completed order eligible for a full refund?" failed because the relevant context was split across a chunk boundary, causing the LLM to miss the exact rule.

### 2. Agent Scenario Results
| Scenario | Status | Model Calls | Tokens Used |
|---|---|---|---|
| S1 | REFUND (128.25) | 5 | 882 |
| S2 | REFUND (135.00) | 5 | 882 |
| S3 | ESCALATE (1999.00) | 5 | 869 |
| S4 | DENIED (0.00) | 4 | 593 |
| S5 | DENIED (0.00) | 3 | 328 |
| S6 | NOT_FOUND (0.00) | 2 | 66 |
| S7 | FALLBACK_MAX_ITERATIONS | 6 | 2982 |

### 3. Stop Safely & Resume Demonstrations
* **Timeout & Token Limits:**
  * Setting `token_limit=200` on S1 successfully triggered `FALLBACK_TOKEN_LIMIT`.
  * Setting `time_limit_seconds=0` on S1 successfully triggered `FALLBACK_TIMEOUT`.
* **Crash-and-Resume:**
  * A simulated crash was injected after Step 2 for S1. 
  * `resume_agent` successfully loaded the state and finished the execution. 
  * The exact sequence of tools executed across the crash and resume was: `get_order`, `search_policy`, `calculate_refund`, `issue_refund`.

### 4. ReAct vs Plan-and-Execute
In our ReAct setup, the agent thinks and acts step-by-step. For S1 (REFUND), it took 5 calls; for S6 (NOT_FOUND), it took just 2 calls. ReAct is highly adaptable—when S6 encountered a missing order, it immediately stopped. However, ReAct is token-heavy and slower because every single step requires a round-trip to the LLM. 

A Plan-and-Execute design would ask the LLM to write the whole 4-step plan upfront, and a Python engine would run it. This takes fewer LLM calls (often just 1 or 2) and fewer tokens. However, if an unexpected observation occurs (like a cancelled order or a system error), the plan breaks and requires an expensive re-planning call. 

**Conclusion:** For this refund workflow, I would choose **ReAct** (or a constrained Graph-based ReAct like LangGraph). Refund policies have complex branching logic (if >45 mins, if >1000 INR). ReAct handles these dynamic conditional branches perfectly by seeing the real output of `calculate_refund` before deciding whether to call `issue_refund` or `escalate_to_human`.

## Limitations and Next Steps
* The Premium flag is a post-hoc whole-window label, making the retention list illustrative rather than a strict leakage-free evaluation.