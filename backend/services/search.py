import os
from backend.database import get_collection
from rank_bm25 import BM25Okapi

def hybrid_search(query: str, limit: int = 3) -> list:
    """
    Retrieves chunks matching the user's query using combined vector and BM25 keywords ranking.
    """
    try:
        collection = get_collection()
        if collection is None:
            return []
            
        # 1. Fetch a broader pool of candidates using vector search
        results = collection.query(query_texts=[query], n_results=limit * 3)
        
        if not results or 'documents' not in results or not results['documents'][0]:
            return []
            
        candidate_docs = results['documents'][0]
        
        # 2. Tokenize and apply BM25 reranking over the fetched pool
        tokenized_corpus = [doc.lower().split(" ") for doc in candidate_docs]
        bm25 = BM25Okapi(tokenized_corpus)
        
        tokenized_query = query.lower().split(" ")
        top_docs = bm25.get_top_documents(tokenized_query, candidate_docs, n=limit)
        
        return top_docs
    except Exception as e:
        print(f"Hybrid search adjustment exception: {e}")
        return []