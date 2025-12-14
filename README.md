🤖 Gemini RAG Agent with Memory

AI-powered Retrieval-Augmented Generation (RAG) system built with Google Gemini, LangChain, and ChromaDB.
This project lets users upload documents, build a knowledge base, and ask questions grounded in those documents — all through a clean Streamlit UI.

🌟 Features
✅ RAG Pipeline (Retrieval + Generation)

Extracts knowledge from uploaded documents

Uses semantic search over ChromaDB

Generates accurate, grounded answers using Google Gemini

✅ Document Uploading (Streamlit)

Upload PDF, TXT, DOCX, MD files

Automatic ingestion + re-indexing

Real-time update of vector database

✅ Memory-Enabled Conversation

Uses ConversationSummaryMemory to maintain context across chat sessions.

✅ Google Gemini Integration

Supports:

models/gemini-2.5-flash

models/gemini-2.5-pro

✅ ChromaDB Vector Store

Persistent storage

Automatically rebuilt when new docs are uploaded

✅ HuggingFace Sentence Embeddings

Uses lightweight + high-performance encoder:

all-MiniLM-L6-v2

🏗️ System Architecture

Here is a visual explanation of the full workflow:

                 ┌──────────────────────────┐
                 │      Streamlit UI        │
                 │  - Chat Input            │
                 │  - Document Upload       │
                 └───────────┬──────────────┘
                             │
                Document Upload Event
                             │
                 ┌───────────▼──────────────┐
                 │     Ingestion Pipeline    │
                 │  - Load Documents         │
                 │  - Split Text             │
                 │  - Embeddings (HF)        │
                 │  - Store in ChromaDB      │
                 └───────────┬──────────────┘
                             │
               RAG Query (User Question)
                             │
                 ┌───────────▼──────────────┐
                 │  Retriever (ChromaDB)     │
                 │  Fetch Top-K Relevant     │
                 │  Document Chunks          │
                 └───────────┬──────────────┘
                             │
                 ┌───────────▼──────────────┐
                 │ Google Gemini LLM         │
                 │ - RAG Prompting           │
                 │ - Conversation Memory     │
                 └───────────┬──────────────┘
                             │
                 ┌───────────▼──────────────┐
                 │      Final Response        │
                 └───────────────────────────┘

📁 Project Structure
gemini_rag_agent/
│
├── app/
│   ├── app.py                 # Streamlit frontend
│   ├── utils.py               # Embeddings, Chroma DB helpers
│   ├── rag_chain.py           # RAG chain builder
│   ├── memory.py              # Conversation memory
│   └── ingest.py              # Document ingestion pipeline
│
├── data/                      # User documents
├── storage/chroma/            # Chroma vector DB (ignored in git)
│
├── .gitignore
├── requirements.txt
└── README.md
