from langchain_groq import ChatGroq

from src.config import GROQ_API_KEY


def create_llm():

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model="openai/gpt-oss-20b",
        temperature=0
    )

    return llm