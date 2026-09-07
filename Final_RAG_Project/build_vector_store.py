from src.loader import load_pdfs
from src.chunker import create_chunks
from src.vector_store import create_vector_store


print("=" * 60)
print("BUILDING RAG VECTOR STORE")
print("=" * 60)

# Step 1: Load PDFs
documents = load_pdfs("data/pdfs")

print("\n" + "=" * 60)
print(f"TOTAL DOCUMENTS: {len(documents)}")
print("=" * 60)

# Step 2: Create chunks
chunks = create_chunks(documents)

print(f"TOTAL CHUNKS: {len(chunks)}")

# Step 3: Create FAISS vector store
create_vector_store(chunks)

print("\nRAG vector database is ready!")