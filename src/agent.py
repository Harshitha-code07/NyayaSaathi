"""Multi-step agent: classify -> follow-ups -> retrieve -> urgency -> plan -> draft offer."""
import os, re
from dotenv import load_dotenv
from . import ml, kb
load_dotenv()

DANGER = re.compile(r"\b(hit me|attack\w*|threat\w*|murder|blackmail\w*|rape|abuse|violence|stolen|stole|missing|kill\w*|scared|assault\w*)\b", re.I)
FOLLOW = {
 "RTI": ["Which department or office holds the information?", "What exact documents or details do you need?", "Have you already applied before? If so, when?"],
 "Consumer complaint": ["What did you buy, and from whom?", "How much did you pay, and when?", "Have you complained to the seller in writing yet?"],
 "Tenant-landlord dispute": ["How much deposit did you pay, and when did you vacate?", "Do you have a written rent agreement?", "Have you asked the landlord in writing for the refund?"],
 "Workplace / labour issue": ["What is the exact problem (unpaid wages, termination, other)?", "How long did you work there, and do you have payslips or a letter?", "Have you raised it with HR in writing?"],
}
PLAN = {
 "RTI": ["Identify the correct department and its Public Information Officer.", "Write your questions clearly, one point per question.", "Pay the application fee as per your state or central rules and keep the receipt.", "Wait for the reply (normally 30 days). If there is none, file a first appeal.", "If still unresolved, go to the Information Commission."],
 "Consumer complaint": ["Collect the bill, warranty, chats and photos of the defect.", "Send a written complaint or legal notice to the seller with a clear deadline (usually 15 days).", "Call the National Consumer Helpline (1915) for pre-litigation help.", "If unresolved, file a complaint on e-Daakhil with the District Consumer Commission, within 2 years of the problem."],
 "Tenant-landlord dispute": ["Gather the rent agreement, receipts and proof of deposit and vacating date.", "Send a written demand notice for the deposit with a deadline (usually 15 days).", "Check your state's tenancy or rent control rules.", "If ignored, approach the rent authority, a mediation centre or civil court, or free legal aid (DLSA)."],
 "Workplace / labour issue": ["Collect the offer letter, payslips, bank statements and messages.", "Raise the issue with HR in writing and keep a copy.", "If unresolved, approach the Labour Commissioner or labour office of your state.", "For harassment at work, contact the Internal Committee."],
}
SYSTEM = "You give general legal information for India. Answer ONLY from the provided context. If it does not cover the question, say so and suggest a lawyer or legal aid. Never invent laws or section numbers. Use simple language and under 150 words."

def llm(prompt):
    key = os.getenv("API_KEY")
    if not key: return None
    import requests
    prov, model = os.getenv("LLM_PROVIDER", "anthropic"), os.getenv("MODEL_NAME")
    try:
        if prov == "anthropic":
            r = requests.post("https://api.anthropic.com/v1/messages", headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
                json={"model": model, "max_tokens": 600, "system": SYSTEM, "messages": [{"role": "user", "content": prompt}]}, timeout=40)
            return r.json()["content"][0]["text"]
        r = requests.post(os.getenv("LLM_BASE_URL", "https://api.openai.com/v1") + "/chat/completions", headers={"Authorization": "Bearer " + key},
            json={"model": model, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]}, timeout=40)
        return r.json()["choices"][0]["message"]["content"]
    except Exception:
        return None

def urgency(cat, text):
    m = re.search(r"(\d+|one|two|three|four|five|six)\s*(day|week|month|year)", text.lower())
    days = 0
    if m:
        n = int(m[1]) if m[1].isdigit() else {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}[m[1]]
        days = n * {"day": 1, "week": 7, "month": 30, "year": 365}[m[2]]
    if cat == "Consumer complaint" and days >= 540: return "High", "Consumer complaints are generally due within 2 years; your time window may be closing."
    if cat == "Workplace / labour issue" and "harass" in text.lower(): return "High", "Harassment complaints have short time limits (generally 3 months)."
    if days >= 365: return "High", "The issue is a year old or more; delay can weaken your case."
    if days >= 60: return "Medium", "Several months have passed; act soon and put your demand in writing."
    return "Low", "No time pressure detected, but act early and keep records."

def run(text):
    steps = []
    c = ml.predict(text); cat = c["label"]
    steps.append(f"Classified the issue as '{cat}' ({c['confidence']:.0%} confidence).")
    if cat == "Other / unclear" or DANGER.search(text):
        steps.append("Issue is outside the supported areas or may involve safety; stopped detailed guidance.")
        return {"cat": cat, "conf": c["confidence"], "alts": c["alternatives"], "out_of_scope": True, "steps": steps}
    q = FOLLOW[cat]; steps.append(f"Prepared {len(q)} follow-up questions to collect missing facts.")
    chunks = kb.retrieve(text + " " + cat); steps.append(f"Retrieved {len(chunks)} knowledge-base passages: " + "; ".join(x['source'] for x in chunks))
    u, why = urgency(cat, text); steps.append(f"Assessed urgency: {u}. {why}")
    ctx = "\n".join(f"[{x['source']}] {x['text']}" for x in chunks)
    ans = llm(f"Context:\n{ctx}\n\nUser problem: {text}\n\nExplain the person's position and key rights.")
    steps.append("Wrote explanation with the LLM, grounded in retrieved text." if ans else "No API key set: built the explanation from retrieved passages (offline mode).")
    ans = ans or " ".join(x["text"] for x in chunks[:2])
    steps.append("Built action plan and the matching document draft option.")
    return {"cat": cat, "conf": c["confidence"], "alts": c["alternatives"], "out_of_scope": False, "followups": q, "chunks": chunks,
            "urgency": u, "why": why, "answer": ans, "plan": PLAN[cat], "steps": steps, "offline": not os.getenv("API_KEY")}
