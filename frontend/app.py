import streamlit as st
import requests

# Set page title and layout
st.set_page_config(page_title="Advanced RAG Bot", layout="wide")

# Ensure core message history state exists
if "messages" not in st.session_state:
    st.session_state.messages = []

# =========================================================
# SIDEBAR BLOCK (Left Side controls)
# =========================================================
with st.sidebar:
    st.title("🤖 Advanced RAG Bot")
    st.subheader("Room Management")
    
    # 🧼 Chat History Reset Button
    if st.button("🧹 Clear Conversation History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.subheader("📂 Ingest Knowledge Base")
    
    # Supported files list
    uploaded_file = st.file_uploader(
        "Upload a document (.txt, .md, .pdf, .docx)", 
        type=["txt", "md", "pdf", "docx"], 
        key="rag_file_uploader"
    )
    
    if uploaded_file is not None:
        # Explicit processing action button to prevent automatic request looping on page re-runs
        if st.button("🚀 Process & Ingest Document", use_container_width=True):
            with st.spinner(f"Vectorizing & indexing {uploaded_file.name}..."):
                try:
                    # Dynamically adjust content types for the multipart/form-data parser
                    content_type = "text/plain"
                    if uploaded_file.name.endswith(".pdf"):
                        content_type = "application/pdf"
                    elif uploaded_file.name.endswith(".docx"):
                        content_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), content_type)}
                    
                    # Inter-container network request targeting your FastAPI pipeline
                    response = requests.post("http://127.0.0.1:8000/api/files/upload", files=files)
                    
                    if response.status_code == 200:
                        st.success(f"Successfully loaded {uploaded_file.name} into ChromaDB!")
                    else:
                        st.error(f"Failed to upload: {response.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Could not connect to backend server: {e}")

# =========================================================
# MAIN DASHBOARD AREA (Right Side Workspace)
# =========================================================
st.title("🤖 Advanced RAG Dashboard")
st.caption("Powered by FastAPI, ChromaDB Hybrid BM25, and Blazing-Fast Groq Inference")

# Render active message streams
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User Chat Input Box
if prompt := st.chat_input("Ask a question about your uploaded materials..."):
    
    # Print user message instantly
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.spinner("Thinking..."):
        try:
            payload = {"query": prompt, "history": st.session_state.messages[:-1]}
            res = requests.post("http://127.0.0.1:8000/api/chat/query", json=payload)
            
            if res.status_code == 200:
                data = res.json()
                answer = data.get("answer")
                
                with st.chat_message("assistant"):
                    st.write(answer)
                    
                    # Collapsible diagnostic trace segment 
                    with st.expander("🔍 See RAG pipeline trace metadata"):
                        st.write(f"**Query Rewriter output:** {data.get('rewritten_query')}")
                        st.write("**Sources used (Hybrid BM25 Ranked):**")
                        st.json(data.get("sources_used"))
                        
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                st.error(f"Backend returned error code {res.status_code}: {res.text}")
        except Exception as e:
            st.error(f"Backend API connection error: {e}")