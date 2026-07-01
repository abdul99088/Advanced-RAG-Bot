from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import os
from groq import Groq
from dotenv import load_dotenv

from backend.services.search import hybrid_search
from backend.services.llm_service import GroqLLMService

load_dotenv()

router = APIRouter(prefix="/api/chat", tags=["chat"])
llm_service = GroqLLMService()

# Initialize direct Groq client for the final answer synthesis stage
api_key = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=api_key) if api_key else None

class ChatRequest(BaseModel):
    query: str
    history: List[Dict[str, str]]

@router.post("/query")
async def chat_endpoint(request: ChatRequest):
    if not groq_client:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not configured on the server.")

    try:
        # 1. Use the LLM Service to rewrite the query based on past conversation history
        rewritten_query = llm_service.rewrite_query(request.query, request.history)
        
        # 2. Query your Chroma Vector Database using hybrid search
        # (This returns an empty list if you haven't uploaded files yet)
        retrieved_docs = hybrid_search(rewritten_query)
        
        # 3. Dynamic Prompting: Adjust behavior based on whether documents exist
        if not retrieved_docs:
            system_prompt = (
                "You are an advanced, helpful RAG AI Assistant.\n"
                "Currently, there are no documents loaded into your vector knowledge base.\n"
                "Answer the user's question accurately using your own general knowledge."
            )
            context_str = "None (Using general LLM knowledge fallback)"
        else:
            context_str = "\n---\n".join(retrieved_docs)
            system_prompt = (
                "You are an expert RAG AI Assistant.\n"
                "Answer the user's question using ONLY the verified context snippets provided below.\n"
                "If the context doesn't contain the answer, rely calmly on your general knowledge to help.\n\n"
                f"Verified Context:\n{context_str}"
            )

        # 4. Compile message structure for Groq
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add historical conversation logs to maintain context window continuity
        for msg in request.history[-5:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
            
        # Add the newly rewritten contextual user prompt
        messages.append({"role": "user", "content": request.query})

        # 5. Execute final completion generation on Groq's current production engine
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.3
        )
        
        answer = response.choices[0].message.content.strip()

        # 6. Return response package alongside debugging metadata trace logs
        return {
            "answer": answer,
            "rewritten_query": rewritten_query,
            "sources_used": retrieved_docs if retrieved_docs else ["No vector sources found. General knowledge fallback applied."]
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error code: 400 - {str(e)}")