import os
import json
import time
import sqlite3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mock_llm

# Ensure output directories exist
os.makedirs("outputs", exist_ok=True)
os.makedirs("state", exist_ok=True)

# --- 1. TASK 1: Chunk the documents ---
def chunk_text(text, filename, chunk_size=40, overlap=10):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i:i+chunk_size]
        chunks.append({"source": filename, "text": " ".join(chunk_words)})
        if i + chunk_size >= len(words):
            break
        i += (chunk_size - overlap)
    return chunks

policies_dir = "data/policies"
documents = {}
all_chunks_10 = []
all_chunks_0 = []

for fn in sorted(os.listdir(policies_dir)):
    if fn.endswith(".txt"):
        with open(os.path.join(policies_dir, fn), "r", encoding="utf-8") as f:
            text = f.read()
            documents[fn] = text
            all_chunks_10.extend(chunk_text(text, fn, 40, 10))
            all_chunks_0.extend(chunk_text(text, fn, 40, 0))

print("=== Task 1: Chunking ===")
print(f"Total chunks (overlap 10): {len(all_chunks_10)}")

# --- 2. TASK 2: Vectorise and retrieve ---
def build_retriever(chunks):
    vec = TfidfVectorizer()
    tfidf = vec.fit_transform([c["text"] for c in chunks])
    return vec, tfidf, chunks

vec_10, tfidf_10, chunks_10 = build_retriever(all_chunks_10)
vec_0, tfidf_0, chunks_0 = build_retriever(all_chunks_0)

def retrieve(query, k=3, vec=vec_10, tfidf=tfidf_10, chunks=chunks_10):
    q_vec = vec.transform([query])
    sims = cosine_similarity(q_vec, tfidf)[0]
    top_k = sims.argsort()[-k:][::-1]
    return [{"source": chunks[i]["source"], "score": sims[i], "text": chunks[i]["text"]} for i in top_k]

# --- 3. TASK 3: Augment and answer (RAG) ---
def answer(question, vec=vec_10, tfidf=tfidf_10, chunks=chunks_10):
    retrieved = retrieve(question, k=3, vec=vec, tfidf=tfidf, chunks=chunks)
    context = ""
    for c in retrieved:
        context += f"[{c['source']}]\n{c['text']}\n"
    
    messages = [
        {"role": "system", "content": "POLICY ASSISTANT. Answer only from the supplied context. If you cannot find the information in the context, say exactly: I could not find that information in the policy documents."},
        {"role": "user", "content": f"CONTEXT:\n{context}\nQUESTION:\n{question}"}
    ]
    
    mock_llm.reset()
    resp = mock_llm.chat(messages)
    return resp.choices[0].message.content, retrieved

# --- 4. TASK 4: Evaluate retrieval and grounding ---
def eval_rag(vec, tfidf, chunks, name):
    eval_df = pd.read_csv("data/raw/policy_eval.csv", keep_default_na=False)
    ans_correct = 0
    doc_correct = 0
    unans_correct = 0
    failed_questions = []

    for _, row in eval_df.iterrows():
        q, exp_doc, exp_fact = row['question'], row['expected_doc'], row['expected_fact']
        ans_text, retrieved = answer(q, vec=vec, tfidf=tfidf, chunks=chunks)
        
        if exp_doc != "NONE":
            sources = [c["source"] for c in retrieved]
            if exp_doc in sources:
                doc_correct += 1
            if exp_fact in ans_text:
                ans_correct += 1
            else:
                failed_questions.append(q)
        else:
            if "I could not find" in ans_text:
                unans_correct += 1

    print(f"\n=== Task 4: Evaluation ({name}) ===")
    print(f"Answerable Questions: Expected Doc in Top 3 = {doc_correct}/12")
    print(f"Answerable Questions: Expected Fact in Answer = {ans_correct}/12")
    print(f"Unanswerable Questions: Fallback Triggered = {unans_correct}/2")
    if failed_questions:
        print(f"Failed Questions: {failed_questions}")

eval_rag(vec_10, tfidf_10, chunks_10, "Overlap 10")
eval_rag(vec_0, tfidf_0, chunks_0, "Overlap 0")

# --- 5. TASK 5: Define Tools ---
def get_order(order_id):
    conn = sqlite3.connect("data/instamart.db")
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id=?", (order_id,))
    row = cur.fetchone()
    conn.close()
    if row: return dict(row)
    return {"error": "order not found"}

def search_policy(query):
    return {"results": retrieve(query, vec=vec_10, tfidf=tfidf_10, chunks=chunks_10)}

def calculate_refund(order_amount, delivery_minutes):
    try:
        amt, mins = float(order_amount), float(delivery_minutes)
    except:
        return {"refund_inr": 0}
    return {"refund_inr": amt if mins > 45 else 0}

def log_ledger(tool, order_id, amt):
    with open("outputs/refund_ledger.jsonl", "a") as f:
        f.write(json.dumps({"timestamp": time.time(), "tool": tool, "order_id": order_id, "amount": amt}) + "\n")

def issue_refund(order_id, refund_inr):
    log_ledger("issue_refund", order_id, refund_inr)
    return {"refund_inr": refund_inr}

def escalate_to_human(order_id, refund_inr):
    log_ledger("escalate_to_human", order_id, refund_inr)
    return {"refund_inr": refund_inr}

TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human
}

# --- 6. TASK 6, 7 & 8: ReAct Agent ---
def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10, run_id=None, crash_after_step=None):
    if run_id is None:
        run_id = f"run_{scenario_id}_{int(time.time())}"
    
    start_time = time.monotonic()
    messages = [
        {"role": "system", "content": "AGENT. You are a refund processing agent."},
        {"role": "user", "content": user_message}
    ]
    tokens_used = 0
    executed_tools = []
    
    mock_llm.reset()
    
    for step in range(1, max_iterations + 1):
        if time.monotonic() - start_time >= time_limit_seconds:
            return {"status": "FALLBACK_TIMEOUT", "final_answer": "Agent timed out.", "steps": executed_tools, "llm_calls": step-1, "tokens_used": tokens_used}
            
        resp = mock_llm.chat(messages)
        t_used = resp.usage.total_tokens
        if tokens_used + t_used >= token_limit:
            return {"status": "FALLBACK_TOKEN_LIMIT", "final_answer": "Agent exceeded token budget.", "steps": executed_tools, "llm_calls": step-1, "tokens_used": tokens_used}
            
        tokens_used += t_used
        msg = resp.choices[0].message
        
        if getattr(msg, 'tool_calls', None):
            # Append assistant message using vars() instead of dict()
            messages.append({"role": "assistant", "content": None, "tool_calls": [vars(tc) for tc in msg.tool_calls]})
            
            tc = msg.tool_calls[0]
            name = tc.name
            args = tc.arguments if isinstance(tc.arguments, dict) else json.loads(tc.arguments)
            
            # Execute tool
            obs = TOOL_MAP[name](**args)
            executed_tools.append(name)
            messages.append({"role": "tool", "tool_call_id": tc.id, "name": name, "content": json.dumps(obs)})
            
            # Save state
            with open(f"state/{run_id}.json", "w") as f:
                json.dump({"run_id": run_id, "scenario_id": scenario_id, "messages": messages, "tokens_used": tokens_used, "step": step, "executed_tools": executed_tools}, f)
            
            # Simulate Crash
            if crash_after_step and step == crash_after_step:
                return {"status": "CRASHED", "run_id": run_id}
                
        else:
            final_ans = msg.content or ""
            status = final_ans.split(":")[0] if ":" in final_ans else "UNKNOWN"
            return {"status": status, "final_answer": final_ans, "steps": executed_tools, "llm_calls": step, "tokens_used": tokens_used}
            
    return {"status": "FALLBACK_MAX_ITERATIONS", "final_answer": "Too many steps. Escalate to human.", "steps": executed_tools, "llm_calls": max_iterations, "tokens_used": tokens_used}

def resume_agent(run_id, max_iterations=6, token_limit=6000, time_limit_seconds=10):
    with open(f"state/{run_id}.json", "r") as f:
        state = json.load(f)
    
    start_time = time.monotonic()
    messages = state["messages"]
    tokens_used = state["tokens_used"]
    executed_tools = state["executed_tools"]
    step_start = state["step"] + 1
    
    mock_llm.reset()
    
    for step in range(step_start, max_iterations + 1):
        resp = mock_llm.chat(messages)
        tokens_used += resp.usage.total_tokens
        msg = resp.choices[0].message
        
        if getattr(msg, 'tool_calls', None):
            # Fixed here as well
            messages.append({"role": "assistant", "content": None, "tool_calls": [vars(tc) for tc in msg.tool_calls]})
            tc = msg.tool_calls[0]
            name = tc.name
            args = tc.arguments if isinstance(tc.arguments, dict) else json.loads(tc.arguments)
            obs = TOOL_MAP[name](**args)
            executed_tools.append(name)
            messages.append({"role": "tool", "tool_call_id": tc.id, "name": name, "content": json.dumps(obs)})
            
            with open(f"state/{run_id}.json", "w") as f:
                json.dump({"run_id": run_id, "scenario_id": state["scenario_id"], "messages": messages, "tokens_used": tokens_used, "step": step, "executed_tools": executed_tools}, f)
        else:
            final_ans = msg.content or ""
            status = final_ans.split(":")[0] if ":" in final_ans else "UNKNOWN"
            return {"status": status, "final_answer": final_ans, "steps": executed_tools, "llm_calls": (step - step_start + 1), "tokens_used": tokens_used}
            
    return {"status": "FALLBACK_MAX_ITERATIONS", "final_answer": "Escalate to human.", "steps": executed_tools, "llm_calls": max_iterations, "tokens_used": tokens_used}

# --- Execution ---
print("\n=== Task 6: Running Agent Scenarios ===")
scenarios = pd.read_csv("data/raw/agent_scenarios.csv")
# Clear previous run logs
open("outputs/agent_runs.jsonl", "w").close()
open("outputs/refund_ledger.jsonl", "w").close()

for _, row in scenarios.iterrows():
    sid, msg, exp_out, exp_amt = row['scenario_id'], row['message'], row['expected_outcome'], row['expected_refund_inr']
    res = run_agent(msg, sid)
    
    # Log to runs
    with open("outputs/agent_runs.jsonl", "a") as f:
        f.write(json.dumps({"scenario_id": sid, "status": res["status"], "final_answer": res["final_answer"], 
                            "llm_calls": res["llm_calls"], "tokens_used": res["tokens_used"], "steps": res["steps"]}) + "\n")
                            
    print(f"Scenario {sid} -> Status: {res['status']}, Calls: {res['llm_calls']}, Tokens: {res['tokens_used']}")

print("\n=== Task 7: Stop Safely Demos ===")
demo_token = run_agent(scenarios.iloc[0]['message'], "S1", token_limit=200)
print(f"S1 Token Limit=200 Demo -> Status: {demo_token['status']}")

demo_time = run_agent(scenarios.iloc[0]['message'], "S1", time_limit_seconds=0)
print(f"S1 Time Limit=0 Demo -> Status: {demo_time['status']}")

print("\n=== Task 8: Crash and Resume Demo ===")
crash_res = run_agent(scenarios.iloc[0]['message'], "S1", run_id="crash_demo_run", crash_after_step=2)
print(f"Run crashed at step 2. Status: {crash_res['status']}")
resume_res = resume_agent("crash_demo_run")
print(f"Resumed Run Finished! Final Status: {resume_res['status']}, Total Tools Executed: {resume_res['steps']}")