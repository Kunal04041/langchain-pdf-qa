# InsightPDF Pro: Professional Document Intelligence

InsightPDF Pro is a modular, high-performance RAG (Retrieval-Augmented Generation) system designed for deep analysis of PDF documents. It combines local embedding intelligence with high-speed cloud inference to provide precise answers based on document context.

---

## Project Overview

InsightPDF Pro provides a streamlined research experience:

- Intelligent PDF Parsing: Extracts and segments PDF text into semantically meaningful chunks, even from encrypted documents.
- Local Vector Search: Uses state-of-the-art sentence embeddings for ultra-fast, local retrieval without recurring API costs for embedding.
- High-Speed Context Analysis: Leverages the Groq LPU platform for near-instantaneous, context-aware responses.
- Professional UI: A dark-themed Streamlit interface optimized for focused research.
- Modular Architecture: Clean separation between API endpoints, core processing logic, and the user interface.

---

## Technical Stack and Models

### Text Generation (LLM)
- Model: Llama 3.3 70B Versatile
- Provider: Groq
- Role: Acts as the primary reasoning engine that synthesizes retrieved document context into natural language answers.

### Text Embeddings
- Model: all-MiniLM-L6-v2
- Library: HuggingFace (via Sentence-Transformers)
- Dimensions: 384
- Role: Converts document chunks and user queries into high-dimensional vectors. This model runs locally on your machine, ensuring data privacy and zero cost.

### RAG Strategy (Chunking & Retrieval)
- Chunking Method: Recursive Character Splitting
- Chunk Size: 1000 characters
- Chunk Overlap: 100 characters
- Search Algorithm: L2 Distance (FAISS)
- Contextual Retrieval: Top 3 most relevant segments per query

### Vector Storage
- Library: FAISS (Facebook AI Similarity Search)
- Role: An efficient local library for similarity search and clustering of dense vectors.

### Deployment & Frameworks
- UI: Streamlit
- Backend API: FastAPI
- PDF Processing: pypdf (with cryptography support for AES encryption)

---

## Role of LangChain

LangChain is utilized as the orchestration layer to bridge document data with the AI models. Specifically, it handles:

- Text Chunking: Using RecursiveCharacterTextSplitter to ensure text segments maintain semantic coherence while fitting model context windows.
- Data Structuring: Utilizing the LangChain Document object to manage text content and associated metadata.
- Vector Store Integration: Providing the wrapper for FAISS to handle document indexing and similarity-based retrieval.
- Embedding Management: Standardizing the interface for the local HuggingFace embedding model.

---

## Setup Instructions

### 1. Environment Setup
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configuration
Create a .env file in the project root or parent directory:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Running the App

#### Professional Research UI (Streamlit)
```powershell
streamlit run app.py
```

#### Programmatic API (FastAPI)
```powershell
uvicorn src.api.main:app --reload
```

---

## Docker Setup

If you prefer to run the application using Docker, follow these steps:

### 1. Build the Docker Image
```bash
docker build -t pdf-qna .
```

### 2. Run the Container
```bash
docker run -p 8502:8501 pdf-qna
```

### 3. Using Docker Compose (Recommended)
```bash
docker-compose up --build
```
The application will be accessible at `http://localhost:8502`.

> **Note:** We use port `8502` for this app to avoid conflict with other Streamlit apps (like TalentScout Pro) which typically use `8501`.

### 4. Stopping & Cleaning Up

*   **To stop:** Press `Ctrl + C` or run `docker-compose down`.
*   **To reclaim disk space:**
    ```powershell
    docker system prune -a --volumes
    ```

---

## Model Resilience

The system includes error handling to suppress technical API logs. If the configured LLM provider is unavailable or rate-limited, the system provides professional fallback communication to the user instead of displaying raw JSON errors.
