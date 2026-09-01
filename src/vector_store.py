from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def create_vector_store(chunks):

    print("\nLoading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print(f"Creating embeddings for {len(chunks)} chunks...")
    print("This may take some time on CPU.\n")

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    vector_store.save_local("data/vector_store")

    print("\n" + "=" * 60)
    print("FAISS VECTOR STORE CREATED SUCCESSFULLY!")
    print("=" * 60)