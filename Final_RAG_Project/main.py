from src.loader import load_pdfs
from src.chunker import create_chunks
from src.vector_store import create_vector_store


documents = load_pdfs("data/pdfs")

print("\n" + "=" * 60)
print(f"TOTAL DOCUMENTS: {len(documents)}")
print("=" * 60)

chunks = create_chunks(documents)

print(f"TOTAL CHUNKS: {len(chunks)}")

create_vector_store(chunks)