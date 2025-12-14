🚀 Gemini RAG Agent with Memory

A Production-Ready Retrieval-Augmented Generation (RAG) System using Google Gemini, LangChain, ChromaDB & Streamlit

📌 Overview

This project implements a fully functional RAG pipeline using:

Google Gemini (2.5 Flash / Pro)

LangChain for orchestration

Sentence Transformers for text embeddings

ChromaDB as the vector store

Streamlit for frontend UI

Conversation Memory for long chats

The agent can:

✔️ Accept document uploads
✔️ Index & embed documents automatically
✔️ Retrieve relevant chunks
✔️ Answer questions grounded in your documents
✔️ Maintain conversation context
✔️ Provide traceable sources

Perfect for interviews, portfolios, real-world use, and learning RAG deeply.

🧠 Key Features
🔍 RAG (Retrieval-Augmented Generation)

Your questions are answered using your documents—not just the model’s training.

📁 Document Upload in UI

Upload pdf, docx, txt, or md files directly in Streamlit.

🧩 Automatic Ingestion Pipeline

On upload, the system:

Loads the document

Splits text into chunks

Embeds using Sentence Transformers

Stores vectors in ChromaDB

💬 Conversation Memory

The agent remembers previous messages using ConversationSummaryMemory.

⚡ Works Offline After Indexing

ChromaDB persists locally.

🧱 Clean Modular Code (Best Practices)

Separate modules:

utils.py → embeddings + vector store
ingest.py → ingestion pipeline
rag_chain.py → RAG chain builder
memory.py → chat memory
app.py → Streamlit UI

🏗️ System Architecture
          ┌────────────────────────────────┐
          │          Streamlit UI          │
          │  • Chat Box                    │
          │  • Document Upload             │
          └────────────────────────────────┘
                          │
                          ▼
             ┌─────────────────────────┐
             │    Ingestion Pipeline   │
             │  • Load documents       │
             │  • Split text           │
             │  • Generate embeddings  │
             │  • Store in ChromaDB    │
             └─────────────────────────┘
                          │
                          ▼
           ┌──────────────────────────────┐
           │     Vector Retriever (K)     │
           │  Fetch relevant document     │
           │  chunks from ChromaDB        │
           └──────────────────────────────┘
                          │
                          ▼
          ┌────────────────────────────────┐
          │        Gemini LLM (2.5)        │
          │  • Combines retrieved chunks   │
          │  • Produces grounded answers   │
          │  • Maintains conversation mem  │
          └────────────────────────────────┘
                          │
                          ▼
                ┌───────────────────┐
                │   Final Answer    │
                │  + Source Citations│
                └───────────────────┘

📂 Project Structure
gemini_rag_agent/
│
├── app/
│   ├── app.py              # Streamlit frontend
│   ├── utils.py            # Embeddings, ChromaDB utils
│   ├── rag_chain.py        # RAG chain builder
│   ├── memory.py           # Conversation memory
│   ├── ingest.py           # Ingestion pipeline
│
├── data/                   # Uploaded documents (ignored in Git)
│
├── storage/chroma/         # Vector DB (ignored in Git)
│
├── .env.example            # Sample environment vars
├── .gitignore              # Ignore sensitive files & local DB
├── requirements.txt        # Dependencies
├── README.md               # Project documentation

⚙️ Setup Instructions
1️⃣ Clone repository
git clone https://github.com/akasherinhikkal-code/gemini_rag_agent.git
cd gemini_rag_agent

2️⃣ Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Configure API Key

Create .env file:

GOOGLE_API_KEY=YOUR_KEY_HERE


Get your key from:
https://aistudio.google.com/apikey

5️⃣ Run document ingestion (optional)

If you placed documents in /data manually:

python app/ingest.py


Otherwise, you can upload them directly from UI.

6️⃣ Start the Streamlit app
streamlit run app/app.py


👉 Opens at:
http://localhost:8501

🧪 Sample Query Flow

Upload a PDF → Ask:

👉 "Summarize the document."
👉 "What are the key sections?"
👉 "Compare chapter 2 and chapter 5."

The agent replies with document-grounded answers + sources.

🎯 Use Cases

✔️ Resume-based Q&A
✔️ Company policy search
✔️ Legal document summarization
✔️ Technical manual Q&A
✔️ Finance reports question answering
✔️ Student study assistant
✔️ Private knowledge-base chatbot

🛡️ Security

.env, /data, /storage are all ignored in GitHub

No personal keys or info will be committed

ChromaDB persists locally only

🛠️ Future Improvements

📌 Add session-based private indices
📌 Support for images + OCR ingestion
📌 Cloud vector database (Pinecone / Weaviate)
📌 Prompt caching
📌 Multi-document analytics

🤝 Contributing

Pull requests are welcome!
If you want new features, feel free to open an issue.