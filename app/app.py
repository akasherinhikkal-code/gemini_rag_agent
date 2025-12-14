import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage
from memory import build_summary_memory
from rag_chain import build_rag_chain
from utils import load_vectorstore, PERSIST_DIR

# Load .env variables
load_dotenv(override=True)

# Configure Streamlit page
st.set_page_config(page_title="Gemini RAG Agent", page_icon="🤖", layout="wide")
st.title("🤖 Gemini RAG Agent with Memory")
st.caption("LangChain · SentenceTransformer + ChromaDB · Streamlit")

# --- GOOGLE API KEY SETUP ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("❌ GOOGLE_API_KEY not found. Create one at https://aistudio.google.com/app/apikey and add it to your .env")
    st.stop()

try:
    genai.configure(api_key=GOOGLE_API_KEY)
    st.sidebar.success(f"✅ Gemini API key loaded: {GOOGLE_API_KEY[:4]}...{GOOGLE_API_KEY[-4:]}")
except Exception as e:
    st.error(f"Failed to configure Gemini API key: {e}")
    st.stop()

# --- Sidebar controls ---
with st.sidebar:
    st.subheader("Model Settings")
    model = st.selectbox("Gemini model", ["models/gemini-2.5-flash", "models/gemini-2.5-pro"], index=0)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    top_k = st.slider("Retriever Top-K", 1, 10, 4, 1)
    st.divider()
    if st.button("Clear chat + summary memory"):
        st.session_state.clear()
        st.experimental_rerun()
    st.divider()

    st.markdown("**Index status**")
    exists = os.path.isdir(PERSIST_DIR) and any(os.scandir(PERSIST_DIR))
    st.write("Chroma DB:", "✅ Found" if exists else "❌ Not found")
    st.caption("Upload documents below to update your RAG database.")

    # 📤 Upload new documents
    st.subheader("📤 Upload New Document")
    uploaded_file = st.file_uploader("Upload a document", type=["pdf", "txt", "docx", "md"])
    if uploaded_file:
        save_path = os.path.join("data", uploaded_file.name)
        os.makedirs("data", exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.read())
        st.success(f"✅ Uploaded {uploaded_file.name} to data/ folder")

        from utils import load_documents, build_vectorstore
        with st.spinner("🔄 Re-indexing documents..."):
            docs = load_documents()
            build_vectorstore(docs)
        st.success("🧠 Chroma DB updated successfully! You can now query the new document.")

# --- LLM + Memory + Chain ---
if "llm" not in st.session_state:
    st.session_state.llm = GoogleGenerativeAI(model=model, temperature=temperature)

try:
    vectorstore = load_vectorstore()
except Exception as e:
    st.error(f"🧠 Failed to load Chroma DB. Run ingestion or upload a document first.\n\nError: {e}")
    st.stop()

if "memory" not in st.session_state:
    st.session_state.memory = build_summary_memory(st.session_state.llm)

if "chain" not in st.session_state:
    st.session_state.chain = build_rag_chain(st.session_state.llm, vectorstore, st.session_state.memory)

# --- Chat UI ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant"):
        st.markdown(msg.content)

user_input = st.chat_input("Ask anything about your documents...")
if user_input:
    st.session_state.messages.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = st.session_state.chain.invoke({"question": user_input})
            answer = result.get("answer", "")
            sources = result.get("source_documents", []) or []
            st.markdown(answer)

            if sources:
                st.markdown("---")
                st.markdown("**Sources**")
                for i, d in enumerate(sources, 1):
                    meta = d.metadata or {}
                    name = meta.get("source") or meta.get("file_path") or f"doc_{i}"
                    st.caption(f"[{i}] {name} (p.{meta.get('page', 'n/a')})")

    st.session_state.messages.append(AIMessage(content=answer))
