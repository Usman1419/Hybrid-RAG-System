from src.llm import create_llm
from src.retriever import create_retriever


def create_rag_pipeline():

    llm = create_llm()
    retriever = create_retriever()

    return llm, retriever


def ask_question(question, llm, retriever):

    documents = retriever.invoke(question)

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    prompt = f"""
Answer the question using only the provided context.

If the answer is not present in the context, say:
"I don't know based on the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    return response.content