
# Advanced Hybrid RAG System
## 🧠 Core Technologies Used
* **FAISS (Facebook AI Similarity Search):** A highly optimized local vector database used for blazing-fast semantic search.
* **BM25 (Best Matching 25):** An advanced information retrieval algorithm used for exact keyword matching.
* **Cross-Encoder Reranker (`ms-marco-MiniLM-L-6-v2`):** Acts as an AI examiner to deeply evaluate and rerank the combined results from FAISS and BM25, ensuring the absolute most relevant chunks are sent to the LLM.
* **Groq API (`openai/gpt-oss-20b` / `Llama 3`):** Powered by LPUs (Language Processing Units), Groq provides lightning-fast inference for generating the final concise answer.
* **HuggingFace Embeddings (`all-MiniLM-L6-v2`):** A fast and local embedding model that converts chunks of text into 384-dimensional vectors.
* **LangChain:** The framework that ties the ingestion, retrieval, and generation pipelines together.

## ⚙️ Architecture Workflow

The system is strictly divided into two distinct phases:

### Phase 1: Database Building (`main.py`)
*This phase is run **only once** to ingest data.*
1. **Loading:** `PyMuPDF` reads all PDF files from the `data/pdfs/` directory.
2. **Chunking:** `RecursiveCharacterTextSplitter` breaks the large documents into smaller chunks (1200 chars, 240 overlap).
3. **Embedding & Storage:** The HuggingFace model converts these text chunks into dense vectors. FAISS then saves these vectors permanently to the hard drive in a binary format (`data/vector_store/index.faiss` and `index.pkl`).

### Phase 2: Question Answering (`query.py`)
*This phase is run daily for interactions.*
1. **Hybrid Retrieval:** When a user asks a question, the system queries FAISS (for semantic meaning) and BM25 (for exact keywords).
2. **Deduplication & Reranking:** The results are combined, duplicates are removed, and the Cross-Encoder scores and reranks the candidates to find the most relevant context.
3. **Generation:** The Top 3 most relevant chunks are injected into a strict prompt template. The LLM processes this context and outputs a concise answer. Crucially, the LLM is explicitly instructed to cite the **exact PDF Source, Page Number, and Relevance Score** at the end of its answer, allowing end-users to fully validate the information.

## 🚀 Setup & Installation

1. **Clone the repository** and open the project directory.
2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Environment Variables:**
   Rename `.env.example` to `.env` and add your Groq API key:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

## 💻 Usage

**Step 1: Build the Vector Store (Run once)**
```bash
python main.py
```
*Note: This process converts all PDFs into vectors. It may take 15-20 minutes on a CPU depending on the size of your PDFs.*

**Step 2: Ask Questions (Run daily)**
```bash
python query.py
```
Type your question when prompted. Type `exit` or `quit` to stop the program.
