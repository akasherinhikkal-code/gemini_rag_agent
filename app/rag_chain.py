from typing import Optional
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

_SYS_PROMPT = """You are a helpful AI assistant. Use the provided context to answer.
- Cite sources briefly like [1], [2] using the document metadata if possible.
- If context is insufficient, say so and ask for a relevant file to be added.
- Keep answers concise and factual.
"""

def make_qa_prompt():
    template = """{system}

Context:
{context}

Conversation so far:
{chat_history}

User question:
{question}

Helpful answer:"""
    return PromptTemplate(
        input_variables=["system", "context", "chat_history", "question"],
        template=template
    )

def build_rag_chain(llm, vectorstore, memory=None) -> ConversationalRetrievalChain:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    qa_prompt = make_qa_prompt()
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": qa_prompt.partial(system=_SYS_PROMPT)},
        return_source_documents=True,
        verbose=False
    )
    return chain
