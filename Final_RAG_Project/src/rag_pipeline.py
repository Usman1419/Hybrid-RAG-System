from src.llm import create_llm
from src.retriever import create_retriever


def create_rag_pipeline():

    llm = create_llm()
    retriever = create_retriever()

    return llm, retriever


def ask_question(question, llm, retriever):

    # Get top 3 most relevant chunks
    documents = retriever.invoke(question)[:3]

    # Inject Metadata (Source, Page, Score) into the context for LLM
    context_parts = []
    for i, doc in enumerate(documents):
        source = doc.metadata.get('source', 'Unknown')
        page = doc.metadata.get('page', 'Unknown')
        score = doc.metadata.get('rerank_score', 0.0)
        
        context_parts.append(
            f"[Chunk {i+1}]\n"
            f"Source PDF: {source}\n"
            f"Page: {page}\n"
            f"Relevance Score: {score:.2f}\n"
            f"Content:\n{doc.page_content}"
        )
    
    context = "\n\n".join(context_parts)

    prompt = f"""
Answer the question using only the provided context.

Rules:
1. Give a direct, professional, and concise answer.
2. Keep your answer brief (maximum 3 to 4 sentences) unless the user explicitly asks for detailed information.
3. If the answer is not present in the context, say: "I don't know based on the provided documents."
4. MUST INCLUDE CITATIONS: At the end of your answer, you MUST explicitly cite the "Source PDF", "Page", and "Relevance Score" from which you derived your answer so the user can validate it. (e.g., "Source: book.pdf, Page: 15, Relevance Score: 8.5")

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    return response.content, documents