from typing import List
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

class HybridRerankRetriever(BaseRetriever):
    vector_store: FAISS = Field(description="FAISS vector store")
    bm25: BM25Okapi = Field(description="BM25 index")
    documents: List[Document] = Field(description="List of all documents")
    reranker: CrossEncoder = Field(description="CrossEncoder reranker")
    k: int = Field(default=40, description="Top K for initial retrieval")
    top_k: int = Field(default=15, description="Top K for final reranked results")

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        # 1. FAISS Search
        faiss_results = self.vector_store.similarity_search_with_score(query, k=self.k)
        
        # 2. BM25 Search
        query_tokens = tokenize_text(query)
        bm25_scores = self.bm25.get_scores(query_tokens)
        bm25_indexes = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:self.k]
        
        # 3. Combine Candidates (Deduplication)
        candidate_docs = {}
        for doc, _ in faiss_results:
            key = (doc.metadata.get("source"), doc.metadata.get("page"), doc.page_content)
            candidate_docs[key] = doc
            
        for idx in bm25_indexes:
            doc = self.documents[idx]
            key = (doc.metadata.get("source"), doc.metadata.get("page"), doc.page_content)
            candidate_docs[key] = doc
            
        candidates = list(candidate_docs.values())
        
        if not candidates:
            return []

        # 4. Rerank with CrossEncoder
        pairs = [(query, doc.page_content) for doc in candidates]
        rerank_scores = self.reranker.predict(pairs)
        
        # Combine doc with its score
        scored_candidates = list(zip(candidates, rerank_scores))
        
        # Sort by score descending
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Return top K docs
        return [doc for doc, score in scored_candidates[:self.top_k]]

import string

def tokenize_text(text: str) -> List[str]:
    # Remove punctuation and split
    translator = str.maketrans('', '', string.punctuation)
    return text.lower().translate(translator).split()

def create_retriever():
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Loading FAISS vector store...")
    vector_store = FAISS.load_local(
        "data/vector_store",
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    print("Loading BM25 index...")
    documents = list(vector_store.docstore._dict.values())
    tokenized_documents = [tokenize_text(doc.page_content) for doc in documents]
    bm25 = BM25Okapi(tokenized_documents)
    
    print("Loading Reranker model...")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    retriever = HybridRerankRetriever(
        vector_store=vector_store,
        bm25=bm25,
        documents=documents,
        reranker=reranker,
        k=40,
        top_k=15
    )

    print("Hybrid Retriever (FAISS + BM25 + Reranker) created successfully.")
    return retriever