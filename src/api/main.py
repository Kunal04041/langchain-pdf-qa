from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import shutil
import os
from src.core.pdf_processor import PDFProcessor
from src.core.vector_store import VectorStoreManager
from src.core.llm import LLMService

app = FastAPI(title="Professional PDF Q&A API")

# Global instances (in a real app, these would be in a lifespan or dependency)
processor = PDFProcessor()
vector_manager = VectorStoreManager()
llm_service = LLMService()

class QuestionRequest(BaseModel):
    question: str

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    
    # Save temp file
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Process and index
        documents = processor.process_pdf(temp_path)
        vector_manager.create_store(documents)
        return {"message": f"Successfully indexed {len(documents)} chunks from {file.filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    if not vector_manager.store:
        raise HTTPException(status_code=400, detail="No PDF has been uploaded and indexed yet.")
    
    # Search for context
    docs = vector_manager.similarity_search(request.question)
    context = "\n---\n".join([d.page_content for d in docs])
    
    # Generate answer
    answer = llm_service.get_answer(context, request.question)
    
    return {
        "answer": answer,
        "sources": [d.page_content[:100] + "..." for d in docs]
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}
