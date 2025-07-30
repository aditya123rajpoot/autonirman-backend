import os
from loguru import logger
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

_llm = None

def get_llm():
    global _llm
    if _llm:
        return _llm

    groq_key = os.getenv("GROQ_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if groq_key:
        try:
            logger.info("✅ Using Groq model: llama3-70b-8192")
            _llm = ChatGroq(
                api_key=groq_key,
                model_name="llama3-70b-8192",
                temperature=0.7
            )
            return _llm
        except Exception as e:
            logger.warning(f"⚠️ Groq init failed: {e} — falling back to OpenAI")

    if openai_key:
        try:
            logger.info("🔁 Using OpenAI GPT-4 fallback")
            _llm = ChatOpenAI(
                openai_api_key=openai_key,  # ✅ Correct param
                model="gpt-4",
                temperature=0.7
            )
            return _llm
        except Exception as e:
            logger.error(f"❌ Fallback GPT-4 failed: {e}")

    raise RuntimeError("❌ No valid API key found for Groq or OpenAI")
