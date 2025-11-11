# Custom AI Agent with Memory (RAG-based) — Gemini Edition

**Tech**: Python · LangChain · Streamlit · Gemini (Google Generative AI) · SentenceTransformer + ChromaDB

This starter replaces OpenAI with **Gemini** while keeping your RAG + long‑term memory flow.
It uses:
- `ChatGoogleGenerativeAI` for the LLM
- `SentenceTransformer` embeddings + `ChromaDB` for retrieval
- `ConversationalRetrievalChain` + `ConversationSummaryMemory` for context + long-term memory
- A minimal Streamlit chat UI

---

## 1) Setup

1. **Create API key** (free/student tier): https://aistudio.google.com/app/apikey
2. **Local env**
   ```bash
   python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   cp .env.example .env
   # paste your key into .env
   ```
3. **Prepare documents**
   - Drop PDFs, TXTs, MDs, DOCXs into the `data/` folder.

4. **Ingest your knowledge base**
   ```bash
   python app/ingest.py
   ```

5. **Run the app**
   ```bash
   streamlit run app/app.py
   ```

---

## 2) Project Structure

```text
gemini_rag_agent/
├── app/
│   ├── app.py                 # Streamlit chat app (Gemini + RAG + memory)
│   ├── rag_chain.py           # RAG chain factory
│   ├── ingest.py              # Build/update Chroma index from /data
│   ├── memory.py              # Long-term memory (summary) wiring
│   └── utils.py               # Helpers (env, loaders, UI bits)
├── data/                      # Put your docs here
├── storage/
│   └── chroma/                # Persistent Chroma DB
├── .streamlit/config.toml
├── .env.example
├── requirements.txt
└── README.md
```

---

## 3) Notes

- Default Gemini model: `gemini-1.5-flash` (fast, low cost). Change in `app/app.py` or via env.
- If you update docs in `data/`, re-run `python app/ingest.py` to refresh the index.
- Streamlit Cloud deploy works on CPU-only.
- For truly *long* dialogs, the summary memory keeps the essence so cost stays low.
