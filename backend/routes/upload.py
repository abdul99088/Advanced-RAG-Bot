import os
from fastapi import APIRouter, UploadFile, File, HTTPException
import chromadb
from backend.database import get_collection  # Assumes this returns your Chroma collection object

router = APIRouter(prefix="/api/files", tags=["files"])

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list:
    """
    Splits long raw text into overlapping windows so the LLM doesn't lose context.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # Simple validation to ensure we are receiving text-based files for now
    if not file.filename.endswith(('.txt', '.md')):
        raise HTTPException(status_code=400, detail="Only plain text (.txt) and markdown (.md) files are supported right now.")

    try:
        # Read the uploaded raw bytes and decode them into a string
        contents = await file.read()
        text_content = contents.decode("utf-8")
        
        # Breakdown into clean paragraphs/chunks
        text_chunks = chunk_text(text_content)
        
        # Connect to your active Chroma collection
        collection = get_collection()
        
        # Prepare metadata, IDs, and payloads for database entry
        documents = []
        metadatas = []
        ids = []
        
        for idx, chunk in enumerate(text_chunks):
            documents.append(chunk)
            metadatas.append({"source": file.filename, "chunk_index": idx})
            ids.append(f"{file.filename}_chunk_{idx}")
            
        # Push everything directly into ChromaDB
        # (Chroma's default embedding function will automatically vectorize the text chunks)
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        return {
            "status": "success",
            "message": f"Successfully parsed '{file.filename}' into {len(text_chunks)} vector chunks!"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion pipeline failed: {str(e)}")