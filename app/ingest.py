import os
from utils import load_documents, build_vectorstore, PERSIST_DIR, DATA_DIR

if __name__ == "__main__":
    if not os.path.isdir(DATA_DIR):
        raise SystemExit(f"No data directory found at {DATA_DIR}")
    docs = load_documents()
    if not docs:
        raise SystemExit("No documents found in data/. Add PDFs, TXTs, MDs, DOCXs and re-run.")
    vs = build_vectorstore(docs)
    print(f"Ingestion complete. Chroma DB persisted to {PERSIST_DIR}")
