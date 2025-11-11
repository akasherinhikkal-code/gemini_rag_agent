import os
from dotenv import load_dotenv
from typing import List
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

PERSIST_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "storage", "chroma")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

def load_env():
    load_dotenv(override=True)
    key = os.getenv("GOOGLE_API_KEY", "")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY is not set. Put it in .env or your environment.")
    return key

def get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    return SentenceTransformerEmbeddings(model_name=model_name)

def build_vectorstore(docs):
    embeddings = get_embedding_model()
    vs = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=PERSIST_DIR)
    vs.persist()
    return vs

def load_vectorstore():
    embeddings = get_embedding_model()
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

def load_documents():
    # Use DirectoryLoader for each type to ensure proper parsing
    docs = []
    patterns = [
        ("**/*.txt", TextLoader),
        ("**/*.md", UnstructuredMarkdownLoader),
        ("**/*.pdf", PyPDFLoader),
        ("**/*.docx", Docx2txtLoader),
    ]
    for pattern, Loader in patterns:
        loader = DirectoryLoader(DATA_DIR, glob=pattern, loader_cls=Loader, show_progress=True, recursive=True)
        try:
            docs.extend(loader.load())
        except Exception as e:
            print(f"Loader failed for pattern {pattern}: {e}")
    return docs
