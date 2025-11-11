from langchain.memory import ConversationSummaryMemory
from langchain.schema.runnable import Runnable
from typing import Optional

def build_summary_memory(llm) -> ConversationSummaryMemory:
    # Keeps a running abstract of the chat. Good for long-term memory.
    return ConversationSummaryMemory(
        llm=llm,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

