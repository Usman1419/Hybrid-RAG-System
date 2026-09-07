import sys
from src.config import GROQ_API_KEY
from src.rag_pipeline import create_rag_pipeline, ask_question

def main():
    print("=" * 60)
    print("INITIALIZING RAG PIPELINE")
    print("=" * 60)
    try:
        llm, retriever = create_rag_pipeline()
    except Exception as e:
        print(f"Failed to initialize pipeline: {e}")
        sys.exit(1)

    print("\nReady! Type 'exit' or 'quit' to stop.\n")

    while True:
        query = input("Ask your question: ").strip()

        if not query:
            print("Please enter a question.\n")
            continue

        if query.lower() in ["exit", "quit"]:
            print("\nGoodbye!")
            break

        print("\nGenerating answer...\n")
        answer, documents = ask_question(query, llm, retriever)
        
        print("=" * 60)
        print("ANSWER")
        print("=" * 60)
        print(answer)
        
        print("\n" + "=" * 60)
        print("SOURCES")
        print("=" * 60)
        
        shown_sources = set()
        count = 0
        
        for doc in documents:
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'Unknown')
            score = doc.metadata.get('rerank_score', 0)
            
            unique_key = (source, page)
            if unique_key not in shown_sources:
                shown_sources.add(unique_key)
                count += 1
                print(f" • {source} (Page {page})  |  Score: {score:.2f}")
                
            if count >= 3:
                break
                
        print("\n")

if __name__ == "__main__":
    main()