import pandas as pd
import re
import json
import random
import time
import mock_llm
from sklearn.metrics import confusion_matrix

# --- 1. Load Data & Initialize ---
tickets = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)
segments = pd.read_csv("outputs/customer_segments.csv")
premium_cids = set(segments[segments['is_premium'] == True]['customer_id'])

# Regex Patterns for PII Redaction
RE_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
RE_PHONE = re.compile(r"\b[6-9]\d{9}\b")
RE_ORDER = re.compile(r"ORD-\d{6}")

ALLOWED_CATS = {"billing", "delivery", "technical", "fraud"}
MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05

def est_tokens(text):
    import math
    return math.ceil(len((text or "").split()) * 1.3)

# --- 2. Redaction & Guardrails ---
def redact_and_guard(row):
    text = row['text']
    if not text.strip():
        return "skipped_empty", "", 0, 0, 0
    
    if est_tokens(text) > MAX_INPUT_TOKENS:
        return "flagged_long", "", 0, 0, 0
    
    # Count and redact
    email_cnt = len(RE_EMAIL.findall(text))
    text = RE_EMAIL.sub("[EMAIL]", text)
    
    phone_cnt = len(RE_PHONE.findall(text))
    text = RE_PHONE.sub("[PHONE]", text)
    
    order_cnt = len(RE_ORDER.findall(text))
    text = RE_ORDER.sub("[ORDER_ID]", text)
    
    return "ok", text, email_cnt, phone_cnt, order_cnt

# Apply Redaction
redact_results = tickets.apply(redact_and_guard, axis=1)
tickets['status'] = [r[0] for r in redact_results]
tickets['clean_text'] = [r[1] for r in redact_results]

# Verification Checkpoint
total_emails = sum(r[2] for r in redact_results)
total_phones = sum(r[3] for r in redact_results)
total_orders = sum(r[4] for r in redact_results)
total_pii = total_emails + total_phones + total_orders
print(f"PII Redacted: {total_emails} emails, {total_phones} phones, {total_orders} orders. Total: {total_pii}")

# Verify no PII is left
remaining_pii = sum(len(RE_EMAIL.findall(t)) + len(RE_PHONE.findall(t)) + len(RE_ORDER.findall(t)) for t in tickets['clean_text'])
print(f"PII left after redaction: {remaining_pii}\n")

# --- 3. Prompts ---
PROMPT_ZERO = """CLASSIFY the following support ticket into exactly one of these categories: billing, delivery, technical, fraud.
Reply strictly in JSON format like this: {"category": "...", "reason": "..."}"""

PROMPT_FEW = """CLASSIFY the following support ticket into exactly one of these categories: billing, delivery, technical, fraud.
Reply strictly in JSON format like this: {"category": "...", "reason": "..."}

EXAMPLES:
User: Someone hacked my account and used my wallet balance. This is fraud, contact [EMAIL].
Assistant: {"category": "fraud", "reason": "User mentioned account hacked and fraud"}

User: The app crashes every time I open the cart.
Assistant: {"category": "technical", "reason": "User reported an app crash"}"""

# --- 4. Resilient LLM Caller ---
def call_llm_with_retry(system_prompt, user_text, ticket_id, budget):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text}
    ]
    
    for n in range(4): # 0, 1, 2, 3 (max 3 retries = 4 attempts)
        try:
            resp = mock_llm.chat(messages)
            tokens = resp.usage.total_tokens
            if budget - tokens < 0:
                return "budget_exhausted", None, tokens
            
            content = resp.choices[0].message.content
            return "success", content, tokens
            
        except mock_llm.TransientLLMError:
            if n == 3: # Failed on last retry
                return "error", None, 0
            wait_time = BASE_DELAY * (2**n) * random.uniform(0.9, 1.15)
            print(f"Retry Log -> Ticket: {ticket_id}, Retry Num: {n}, Wait: {wait_time:.4f}s")
            time.sleep(wait_time)

def validate_output(content):
    try:
        data = json.loads(content)
        cat = data.get("category", "").lower()
        if cat in ALLOWED_CATS:
            return "classified_json", cat
    except json.JSONDecodeError:
        pass
    
    # Fallback keyword extraction
    words = set(re.findall(r"[a-z]+", content.lower()))
    found_cats = words.intersection(ALLOWED_CATS)
    if len(found_cats) == 1:
        return "classified_keyword", list(found_cats)[0]
    
    return "flagged_invalid_output", None

# --- 5. Execution Pipeline ---
def run_pipeline(prompt, budget, save_path=None):
    mock_llm.reset()
    results = []
    tokens_used = 0
    
    for _, row in tickets.iterrows():
        if row['status'] != "ok": # Skip empty/long
            results.append({'ticket_id': row['ticket_id'], 'customer_id': row['customer_id'], 
                            'status': row['status'], 'category': "", 'segment_name': "", 
                            'priority': "NORMAL", 'route': "human_review", 'tokens_used': 0})
            continue
            
        call_status, content, t_used = call_llm_with_retry(prompt, row['clean_text'], row['ticket_id'], budget - tokens_used)
        
        if call_status == "budget_exhausted":
            results.append({'ticket_id': row['ticket_id'], 'customer_id': row['customer_id'], 
                            'status': "budget_exhausted", 'category': "", 'segment_name': "", 
                            'priority': "NORMAL", 'route': "human_review", 'tokens_used': 0})
            continue
            
        tokens_used += t_used
        final_status, final_cat = validate_output(content)
        
        # Routing Logic
        cid = row['customer_id']
        is_prem = cid in premium_cids
        segment_name = "Premium" if is_prem else "" # Simplified for routing
        
        if final_status == "flagged_invalid_output":
            route, priority = "human_review", "NORMAL"
        elif final_cat == "fraud":
            route, priority = "security_team", "HIGH"
        elif is_prem:
            route, priority = "premium_desk", "HIGH"
        else:
            route, priority = "support_queue", "NORMAL"
            
        results.append({'ticket_id': row['ticket_id'], 'customer_id': cid, 
                        'status': final_status, 'category': final_cat or "", 'segment_name': segment_name, 
                        'priority': priority, 'route': route, 'tokens_used': t_used})
        
    df_res = pd.DataFrame(results)
    if save_path:
        df_res.to_csv(save_path, index=False)
    return df_res, tokens_used

# --- 6. Run & Compare ---
print("=== Running Zero-Shot Pipeline (Budget 6000) ===")
zero_df, zero_tokens = run_pipeline(PROMPT_ZERO, 6000, "outputs/triage_budget_demo.csv")
print(f"Budget Exhausted Tickets: {len(zero_df[zero_df['status'] == 'budget_exhausted'])}")

print("\n=== Running Few-Shot Pipeline (Budget 100000) ===")
few_df, few_tokens = run_pipeline(PROMPT_FEW, 100000, "outputs/triage_results.csv")

# Metrics Calculation
def print_metrics(name, df_res, total_t):
    classified_df = df_res[df_res['status'].isin(['classified_json', 'classified_keyword'])]
    merged = classified_df.merge(tickets[['ticket_id', 'true_category']], on='ticket_id')
    acc = (merged['category'] == merged['true_category']).mean()
    print(f"\n{name} Metrics:")
    print(f"Classified Tickets: {len(classified_df)}")
    print(f"Accuracy: {acc * 100:.1f}%")
    print(f"Total Tokens: {total_t}")
    print(f"Tokens per Classified Ticket: {total_t / len(classified_df):.1f}")
    return merged

print("\n--- Status Breakdown (Few-Shot) ---")
print(few_df['status'].value_counts())

zero_merged = print_metrics("Zero-Shot", zero_df[zero_df['status'] != 'budget_exhausted'], zero_tokens)
print_metrics("Few-Shot", few_df, few_tokens)

print("\n=== Zero-Shot Confusion Matrix ===")
labels = ["billing", "delivery", "technical", "fraud"]
cm = confusion_matrix(zero_merged['true_category'], zero_merged['category'], labels=labels)
cm_df = pd.DataFrame(cm, index=[f"True_{l}" for l in labels], columns=[f"Pred_{l}" for l in labels])
print(cm_df)
print("\nSuccess: Part C Triage outputs saved to outputs/triage_budget_demo.csv and outputs/triage_results.csv!")