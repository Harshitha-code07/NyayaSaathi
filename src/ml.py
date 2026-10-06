"""Synthetic dataset + TF-IDF/LogReg issue classifier."""
import random, json, joblib
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

ROOT = Path(__file__).resolve().parent.parent
LABELS = ["RTI", "Consumer complaint", "Tenant-landlord dispute", "Workplace / labour issue", "Other / unclear"]
OPEN = ["", "Please help, ", "Sir, ", "Hi, ", "I need advice. ", "Urgent: "]
DATA = {
 "RTI": (["I want to know", "How can I find out", "I need details of", "Please help me get information about", "I want to ask the government about"],
  ["how the municipal office spent funds on road repair", "the status of my pension application at the government office", "why my ration card application is still pending", "who approved the building permit near my house", "the budget used for the village school", "the tender details of the local bridge project", "how the panchayat used the water scheme money", "the file notings on my scholarship application", "the result of my passport police verification", "which contractors got public works orders in my ward"],
  ["", " in my area", " from the government department", " because the office is not telling me", " as a citizen", " using RTI"]),
 "Consumer complaint": (["I bought", "I ordered online", "I paid for", "Last month I purchased", "I got"],
  ["a phone that stopped working in two weeks and the seller refuses a refund", "a washing machine that is defective and the company will not repair it", "a laptop and received a different model than advertised", "a flight ticket that was cancelled and the airline gives no refund", "a coaching course but the institute refuses to return the fees", "a sofa that arrived damaged and the shop ignores my calls", "an insurance policy but the claim was wrongly rejected", "shoes online and the website refuses to accept my return", "a gym membership but the gym closed down without refund", "a mixer grinder with a warranty but the service centre denies repair"],
  ["", " and I have the bill", " and they are not responding", " and customer care keeps delaying", " and I want my money back", " and the product is faulty"]),
 "Tenant-landlord dispute": (["My landlord", "The house owner", "My owner", "The flat owner", "My landlord and I"],
  ["has not returned my security deposit after I vacated", "is refusing to refund the advance of 30000 rupees", "increased the rent suddenly without any notice", "is asking me to leave the house without notice", "keeps cutting my deposit for no reason", "entered my rented flat without permission", "refuses to give a rent receipt or rent agreement", "is deducting the full deposit for normal wear and tear", "stopped the water supply to force me out", "does not repair leakage in my rented house"],
  ["", " for three months", " even after repeated reminders", " and the agreement is expired", " and I have the rent agreement", " and I vacated last month"]),
 "Workplace / labour issue": (["My employer", "My company", "My boss", "My manager", "The factory owner"],
  ["has not paid my salary for two months", "terminated me suddenly without any notice", "is not giving my full and final settlement", "makes me work overtime without extra pay", "is not paying my provident fund", "refused to issue my relieving letter", "is harassing me at work", "cut my wages without telling me", "denied my leave and bonus", "forced me to resign"],
  ["", " and I have no written contract", " even though I worked for a year", " and HR is ignoring me", " and I need my dues", " and I am a daily wage worker"]),
 "Other / unclear": (["Someone", "I want to", "Can you tell me", "Someone hit me and", "I am confused about"],
  ["stole my wallet on the bus", "attacked me and I am scared", "what is the weather today", "how to cook biryani", "my neighbour threatened to hurt me", "get a divorce and custody", "my property is under a criminal case", "which movie to watch", "I am being blackmailed online", "a murder case in my town", "how to learn cricket", "my friend is missing"],
  ["", " please advise", " not sure what to do", " help me"]),
}

def _noise(s, rng):
    if rng.random() < .12 and len(s) > 8:
        i = rng.randrange(2, len(s) - 2); s = s[:i] + s[i + 1:]
    if rng.random() < .1: s += " bhai"
    return s.lower() if rng.random() < .25 else s

def generate(n=220, seed=7):
    rng = random.Random(seed); rows = []
    for lab, (a, b, c) in DATA.items():
        combos = [(x, y, z) for x in a for y in b for z in c]
        for x, y, z in rng.sample(combos, min(n, len(combos))):
            rows.append((_noise(rng.choice(OPEN) + f"{x} {y}{z}".strip(), rng), lab))
    df = pd.DataFrame(rows, columns=["text", "label"]).sample(frac=1, random_state=1)
    (ROOT / "data").mkdir(exist_ok=True); df.to_csv(ROOT / "data/issues.csv", index=False)
    return df

def train():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    df = generate()
    Xtr, Xte, ytr, yte = train_test_split(df.text, df.label, test_size=.2, stratify=df.label, random_state=42)
    m = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True), LogisticRegression(max_iter=1000, C=8))
    m.fit(Xtr, ytr); pred = m.predict(Xte)
    cm = confusion_matrix(yte, pred, labels=LABELS)
    out = ROOT / "models"; out.mkdir(exist_ok=True); joblib.dump(m, out / "clf.joblib")
    json.dump({"accuracy": accuracy_score(yte, pred), "n_total": len(df), "n_test": len(yte),
               "counts": df.label.value_counts().to_dict(),
               "report": classification_report(yte, pred, output_dict=True)}, open(out / "metrics.json", "w"), indent=1)
    fig, ax = plt.subplots(figsize=(6, 5)); ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(5)); ax.set_yticks(range(5)); ax.set_xticklabels([l.split()[0] for l in LABELS]); ax.set_yticklabels([l.split()[0] for l in LABELS])
    for i in range(5):
        for j in range(5): ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); fig.tight_layout(); fig.savefig(out / "confusion_matrix.png", dpi=140)
    return accuracy_score(yte, pred)

_m = None
def predict(text):
    global _m
    if _m is None:
        if not (ROOT / "models/clf.joblib").exists(): train()
        _m = joblib.load(ROOT / "models/clf.joblib")
    p = _m.predict_proba([text])[0]; order = p.argsort()[::-1]; c = _m.classes_
    return {"label": c[order[0]], "confidence": float(p[order[0]]), "alternatives": [(c[i], float(p[i])) for i in order[1:3]]}

if __name__ == "__main__":
    print("accuracy", train())
