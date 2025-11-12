# 🤖 Gemini RAG Agent with Memory

An intelligent **Retrieval-Augmented Generation (RAG)** application powered by **Google Gemini**, capable of understanding and answering questions from your uploaded documents using **LangChain**, **ChromaDB**, and **HuggingFace embeddings**.

---

## 🧩 Project Overview

**Goal:**  
To create an AI agent that reads multiple documents, builds a searchable knowledge base, and answers user queries grounded in those documents.

**Built With:**  
- 🧠 Google Gemini API (2.5 Flash / Pro)
- 🧬 LangChain Framework
- 📚 Chroma Vector Database
- 🗂️ HuggingFace Sentence Embeddings
- 💾 Streamlit Frontend

---

## ⚙️ System Architecture

### **1️⃣ Document Upload**
Users place their documents (PDFs, TXTs, etc.) into the `/data` folder.  
Each file is read, split into chunks, and processed for embedding.

### **2️⃣ Chunking & Embedding**
Text is split into semantic chunks (e.g., 1000 characters) and embedded into numerical vectors using:
```python
from langchain.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
