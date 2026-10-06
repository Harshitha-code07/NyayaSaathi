# NyayaSaathi - AI legal-information assistant (SDG 16)
1. `pip install -r requirements.txt`
2. (Optional) copy `.env.example` to `.env` and set `API_KEY`, `LLM_PROVIDER`, `MODEL_NAME`. Without a key the app runs in offline mode.
3. `python -m src.ml` (trains the classifier; also auto-runs on first launch)
4. `streamlit run app.py`
Screenshot guide: Home, Assistant result (badges + plan + "What the agent did"), Document Drafts preview, Model Insights.
Verify all legal facts in `src/kb.py` against official sources before real use.
