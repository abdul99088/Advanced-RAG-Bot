import os
import chromadb

# Define a persistent directory path on your E: drive to save vectors
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chroma_db")

# Initialize the persistent Chroma client
chroma_client = chromadb.PersistentClient(path=DB_PATH)

def get_collection(collection_name: str = "rag_documents"):
    """
    Fetches an existing Chroma collection or creates a new one.
    """
    return chroma_client.get_or_create_collection(name=collection_name)