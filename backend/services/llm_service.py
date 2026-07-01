import os
from groq import Groq
from dotenv import load_dotenv

# Ensure your .env variables are loaded
load_dotenv()

class GroqLLMService:
    def __init__(self):
        # Grabs the gsk_ key from your .env file
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set in your environmental parameters!")
        self.client = Groq(api_key=api_key)
        
    def rewrite_query(self, current_query: str, chat_history: list) -> str:
        """
        Looks at the last conversation steps to rewrite ambiguous inputs.
        """
        if not chat_history:
            return current_query
            
        history_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history[-7:]])
        
        prompt = f"""
        Given the following chat logs and a final user input, convert the final input into a standalone, fully-contextual search query.
        Do NOT answer the question. Only output the rewritten search term.

        Logs:
        {history_context}

        Input: {current_query}
        Rewritten Query:"""
        
        # --- FIXED INDENTATION HERE ---
        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # Live Groq production model ID
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return response.choices[0].message.content.strip()