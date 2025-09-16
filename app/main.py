# FILE: app/main.py

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import shutil
import os

from app.processing import load_and_split_document
from app.qa_chain import get_qa_chain_for_candidate
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# --- App Initialization ---
app = FastAPI(
    title="Multi-Candidate Q&A API",
    description="API to upload CVs for different candidates and ask questions about them.",
    version="2.0.0"
)

# --- Database Initialization ---
# This will create a persistent directory for our vector database
CHROMA_PATH = "chroma_db"

# Get the Ollama host from an environment variable, defaulting to localhost
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
OLLAMA_BASE_URL = f"http://{OLLAMA_HOST}:11434"

embeddings = OllamaEmbeddings(model="llama3", base_url=OLLAMA_BASE_URL)
vectorstore = Chroma(persist_directory=CHROMA_PATH,
                     embedding_function=embeddings)

# --- API Models ---


class QuestionRequest(BaseModel):
    question: str


class UploadResponse(BaseModel):
    message: str
    candidate_id: str
    filename: str

# --- API Endpoints ---


@app.post("/upload_cv/{candidate_id}", response_model=UploadResponse)
def upload_cv(candidate_id: str, file: UploadFile = File(...)):
    """
    Uploads a CV for a specific candidate, processes it, and stores it.
    """
    # Create a temporary path to save the file
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Process the document
    splits = load_and_split_document(temp_file_path)

    # Add candidate_id to metadata for each document chunk
    ids = [f"{candidate_id}_{i}" for i, _ in enumerate(splits)]
    for i, split in enumerate(splits):
        split.metadata = {"candidate_id": candidate_id}

    # Add to the vector store
    vectorstore.add_documents(documents=splits, ids=ids)
    print(f"Stored {len(splits)} chunks for candidate {candidate_id}")

    # Clean up the temporary file
    os.remove(temp_file_path)

    return UploadResponse(
        message="CV processed and stored successfully.",
        candidate_id=candidate_id,
        filename=file.filename
    )


@app.post("/ask/{candidate_id}")
def ask_question(candidate_id: str, request: QuestionRequest):
    """
    Asks a question about a specific candidate.
    """
    print(
        f"Received question for candidate {candidate_id}: {request.question}")

    # Check if there are any documents for this candidate
    # This is a simple check; a real app would need a more robust way
    results = vectorstore.get(where={"candidate_id": candidate_id}, limit=1)
    if not results['ids']:
        raise HTTPException(
            status_code=404, detail=f"No documents found for candidate ID '{candidate_id}'. Please upload a CV first.")

    # Create a chain filtered for this candidate and ask the question
    qa_chain = get_qa_chain_for_candidate(vectorstore, candidate_id)
    response = qa_chain.invoke({"input": request.question})

    return {
        "candidate_id": candidate_id,
        "question": request.question,
        "answer": response["answer"]
    }
