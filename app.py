import json, html
from pathlib import Path
import streamlit as st, pandas as pd
from src import agent, docs, ml

st.set_page_config(page_title="NyayaSaathi", page_icon="⚖️", layout="wide")
ROOT = Path(__file__).parent
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;800&display=swap');
html, body, [class*="css"] {font-family:'Manrope',sans-serif;}
#MainMenu, footer, header [data-testid="stToolbar"], .stDeployButton {visibility:hidden;}
.block-container {padding-top:2rem; max-width:1150px;}
.hero {background:#1B2A49; color:#fff; padding:44px 48px; border-radius:6px; border-bottom:5px solid #E8A33D; margin-bottom:24px;}
.hero h1 {color:#fff; font-size:2.7rem; font-weight:800; margin:0 0 8px 0; letter-spacing:-.5px;}
.hero p {color:#D6DCE8; font-size:1.1rem; max-width:640px; margin:0;}
.card {background:#fff; border:1px solid #DDE3EA; border-radius:8px; padding:20px 22px; margin-bottom:14px;}
.card h4 {margin:0 0 6px 0; color:#1B2A49;}
.metric {background:#fff; border:1px solid #DDE3EA; border-left:5px solid #1B2A49; border-radius:6px; padding:14px 18px;}
.metric small {color:#5B6678;} .metric b {display:block; font-size:1.5rem; color:#1B2A49;}
.pill {display:inline-block; padding:3px 12px; border-radius:999px; font-weight:600; font-size:.85rem; color:#fff;}
.Low {background:#2E7D5B;} .Medium {background:#C98A12;} .High {background:#B3362F;}
.chip {display:inline-block; background:#E8EDF4; color:#1B2A49; border-radius:6px; padding:4px 10px; margin:2px 6px 2px 0; font-size:.82rem;}
.paper {background:#fff; border:1px solid #CBD3DD; box-shadow:0 2px 10px rgba(0,0,0,.08); padding:36px 40px; white-space:pre-wrap; font-family:Georgia,serif; line-height:1.55; min-height:520px;}
.warn {background:#FFF6E0; border-left:5px solid #E8A33D; padding:12px 16px; border-radius:4px; margin:10px 0;}
.stButton>button, .stDownloadButton>button {border-radius:6px; font-weight:600;}
[data-testid="stSidebar"] {background:#fff; border-right:1px solid #DDE3EA;}
</style>"""
st.markdown(CSS, unsafe_allow_html=True)
SC = ["My landlord has not returned my 30,000 rupee deposit for 3 months after I vacated.",
      "I want to know how my municipal office spent funds for road repair in my area.",
      "I bought a phone online, it stopped working in 2 weeks and the seller refuses a refund."]
PAGES = ["Home", "Legal Assistant", "Document Drafts", "Model Insights", "About"]
ss = st.session_state
ss.setdefault("page", "Home"); ss.setdefault("result", None); ss.setdefault("facts", {}); ss.setdefault("q", "")

def go(page, q=None):
    ss.page = page
    if q: ss.q = q

with st.sidebar:
    st.markdown("## ⚖️ NyayaSaathi\n*Know your rights. Take the first step.*")
    st.radio("Navigate", PAGES, key="page", label_visibility="collapsed")
    st.markdown('<div class="warn"><b>Not legal advice.</b> NyayaSaathi gives general legal information. For serious matters, consult a qualified lawyer.</div>', unsafe_allow_html=True)
    st.caption("Your inputs stay in this browser session only and are not stored.")

def card(t, b): st.markdown(f'<div class="card"><h4>{t}</h4>{b}</div>', unsafe_allow_html=True)

def home():
    st.markdown('<div class="hero"><h1>Legal help you can understand.</h1><p>Describe your problem in plain words. NyayaSaathi finds the right category, explains your options from verified sources, and drafts the first letter for you.</p></div>', unsafe_allow_html=True)
    c = st.columns(3)
    for col, (t, b) in zip(c, [("Classify", "A trained ML model sorts your problem into RTI, consumer, tenant or workplace."), ("Understand", "Answers come only from a curated legal knowledge base, with sources shown."), ("Draft", "Get a ready-to-edit RTI, legal notice or deposit demand as PDF or Word.")]):
        with col: card(t, b)
    st.markdown("#### Try an example")
    c = st.columns(3)
    for col, s in zip(c, SC):
        with col:
            st.markdown(f'<div class="card" style="min-height:90px">{html.escape(s)}</div>', unsafe_allow_html=True)
            st.button("Ask this", key=s, on_click=go, args=("Legal Assistant", s), use_container_width=True)

def assistant():
    st.title("Legal Assistant")
    st.text_area("Describe your problem", key="q", height=110, max_chars=2000, placeholder="e.g. My landlord has not returned my deposit for 3 months...")
    if st.button("Analyze my issue", type="primary") and ss.q.strip():
        with st.spinner("Understanding your issue... searching legal sources..."):
            ss.result = agent.run(ss.q.strip()); ss.facts = {"details": ss.q.strip()}
    r = ss.result
    if not r: st.info("Describe your problem above, or pick an example from Home."); return
    if r["out_of_scope"]:
        st.markdown('<div class="warn"><b>This looks outside what NyayaSaathi can help with.</b><br>If you are in danger, call <b>112</b> now. For criminal matters, please speak to a lawyer or your District Legal Services Authority (free legal aid, NALSA helpline 15100 - verify before use).</div>', unsafe_allow_html=True)
    else:
        if r["offline"]: st.caption("Running in offline mode (no API key). Answers are built from the knowledge base.")
        a, b, c = st.columns(3)
        a.markdown(f'<div class="metric"><small>Issue type</small><b>{r["cat"]}</b></div>', unsafe_allow_html=True)
        b.markdown(f'<div class="metric"><small>Confidence</small><b>{r["conf"]:.0%}</b></div>', unsafe_allow_html=True)
        c.markdown(f'<div class="metric"><small>Urgency</small><span class="pill {r["urgency"]}">{r["urgency"]}</span><div style="font-size:.8rem;margin-top:4px">{r["why"]}</div></div>', unsafe_allow_html=True)
        st.write("")
        card("What this means for you", html.escape(r["answer"]))
        card("Your action plan", "<ol>" + "".join(f"<li>{html.escape(s)}</li>" for s in r["plan"]) + "</ol>")
        st.markdown("**Sources:** " + "".join(f'<span class="chip">{html.escape(x["source"])}</span>' for x in r["chunks"]), unsafe_allow_html=True)
        st.markdown("#### Help us fill in the details")
        with st.form("facts_form"):
            st.caption("Questions to consider: " + " | ".join(r["followups"]))
            f1, f2 = st.columns(2)
            n = f1.text_input("Your name"); o = f2.text_input("Other party (landlord, seller, office)")
            am = f1.text_input("Amount (if any)"); w = f2.text_input("Date / duration")
            ad = st.text_input("Your address"); d = st.text_area("Facts in your words", value=ss.q)
            if st.form_submit_button("Save and go to draft"):
                ss.facts = {"name": n, "other": o, "amount": am, "when": w, "address": ad, "details": d}
                ss.page = "Document Drafts"; st.rerun()
    with st.expander("What the agent did"):
        for i, s in enumerate(r["steps"], 1): st.write(f"{i}. {s}")

def drafts():
    st.title("Document Drafts")
    default = {"RTI": 0, "Consumer complaint": 1, "Tenant-landlord dispute": 2}.get((ss.result or {}).get("cat"), 0)
    kind = st.selectbox("Document type", docs.KINDS, index=default)
    L, R = st.columns(2); f = dict(ss.facts)
    with L:
        for k, lab in [("name", "Your name"), ("other", "Other party / office"), ("amount", "Amount"), ("when", "Date"), ("address", "Address")]:
            f[k] = st.text_input(lab, value=f.get(k, ""), key="d_" + k)
        f["details"] = st.text_area("Details", value=f.get("details", ""), height=160, key="d_details")
    title, body = docs.build(kind, f)
    with R:
        st.markdown(f'<div class="paper"><b>{html.escape(title)}</b>\n\n{html.escape(body)}\n\n<i style="color:#777;font-size:.8rem">{docs.FOOT}</i></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.download_button("Download PDF", docs.to_pdf(title, body), "draft.pdf", "application/pdf", use_container_width=True)
    c2.download_button("Download Word", docs.to_docx(title, body), "draft.docx", use_container_width=True)

def insights():
    st.title("Model Insights")
    if not (ROOT / "models/metrics.json").exists(): ml.train()
    m = json.load(open(ROOT / "models/metrics.json")); rep = m["report"]
    a, b, c = st.columns(3)
    a.metric("Accuracy (held-out)", f'{m["accuracy"]:.1%}'); b.metric("Training examples", m["n_total"]); c.metric("Test examples", m["n_test"])
    st.markdown('<div class="warn">The dataset is <b>synthetic</b>, so accuracy is optimistic. Next step: validate on real citizen complaints.</div>', unsafe_allow_html=True)
    L, R = st.columns(2)
    with L:
        st.markdown("**Class distribution**"); st.bar_chart(pd.Series(m["counts"]))
        st.markdown("**Classification report**"); st.dataframe(pd.DataFrame({k: v for k, v in rep.items() if k in ml.LABELS}).T.round(2))
    with R:
        st.markdown("**Confusion matrix**"); st.image(str(ROOT / "models/confusion_matrix.png"))
    t = st.text_input("Try the classifier")
    if t: p = ml.predict(t); st.write(f'**{p["label"]}** ({p["confidence"]:.0%}); alternatives: ' + ", ".join(f"{l} ({s:.0%})" for l, s in p["alternatives"]))

def about():
    st.title("About NyayaSaathi")
    card("The problem (SDG 16: Access to Justice)", "Many people do not know their basic rights or how to file an RTI, consumer complaint or deposit notice. Legal help is costly and intimidating.")
    card("How it works", "1. ML classifier (TF-IDF + Logistic Regression) detects the issue type.<br>2. Retriever finds matching passages in the knowledge base.<br>3. Agent assesses urgency, builds an action plan and drafts the document.<br>4. An optional LLM rewrites the explanation using only the retrieved text.")
    card("Limitations and ethics", "Provides general information, not legal advice. Knowledge base is small and must be verified against official sources (indiacode.nic.in, rti.gov.in, consumerhelpline.gov.in). Criminal and safety matters are declined. Classifier is trained on synthetic data. No personal data is stored.")

{"Home": home, "Legal Assistant": assistant, "Document Drafts": drafts, "Model Insights": insights, "About": about}[ss.page]()
