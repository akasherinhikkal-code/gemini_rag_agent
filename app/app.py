import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage
from memory import build_summary_memory
from rag_chain import build_rag_chain
from utils import load_vectorstore, PERSIST_DIR

load_dotenv(override=True)

st.set_page_config(page_title="Gemini RAG Agent", page_icon="🤖", layout="wide")
st.title("🤖 Gemini RAG Agent with Memory")
st.caption("LangChain · SentenceTransformer + ChromaDB · Streamlit")

if not os.getenv("GOOGLE_API_KEY"):
    st.error("GOOGLE_API_KEY is not set. Create one at https://aistudio.google.com/app/apikey and put it in .env")
    st.stop()

# Sidebar controls
with st.sidebar:
    st.subheader("Model Settings")
    model = st.selectbox(
    "Gemini model",
    [
        "models/gemini-2.5-flash",
        "models/gemini-2.5-pro",
        "models/gemini-flash-latest",
        "models/gemini-pro-latest",
    ],
    index=0,
)
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
    st.caption("Run `python app/ingest.py` after putting docs in `data/`.")

# Initialize LLM + vectorstore + memory + chain
if "llm" not in st.session_state:
    st.session_state.llm = GoogleGenerativeAI(model=model, temperature=temperature)
else:
    # update if user changed controls
    st.session_state.llm = GoogleGenerativeAI(model=model, temperature=temperature)

try:
    vectorstore = load_vectorstore()
except Exception as e:
    st.error(f"Failed to load Chroma DB. Did you run ingest? Error: {e}")
    st.stop()

if "memory" not in st.session_state:
    st.session_state.memory = build_summary_memory(st.session_state.llm)

if "chain" not in st.session_state:
    st.session_state.chain = build_rag_chain(st.session_state.llm, vectorstore, st.session_state.memory)

# Chat history
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
