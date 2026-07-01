from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# --- UPDATED TO DIRECT, EXPLICIT IMPORTS ---
from backend.routes.chat import router as chat_router
from backend.routes.upload import router as upload_router

app = FastAPI(title="Advanced RAG Bot Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include both routers using their specific variable names
app.include_router(chat_router)
app.include_router(upload_router)

@app.get("/")
def read_root():
    return {"status": "Backend is running flawlessly"}
