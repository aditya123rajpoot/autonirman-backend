import os
from langchain.chat_models import Groq

_llm = None

def get_llm():
    global _llm
    if not _llm:
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY not set in environment")
        _llm = Groq(
            api_key=key,
            model="groq-mistral-16k",   # or your selected Groq model
            temperature=0.7
        )
    return _llm
