# 🤖 Advanced RAG Dashboard
> **An End-to-End Hybrid Search RAG Pipeline Powered by FastAPI, ChromaDB, and Blazing-Fast Groq Inference.**

[![Live App on Hugging Face](https://img.shields.io/badge/%F0%9F%A5%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/khaliqabdull/humanizer3.0)
[![GitHub Portfolio](https://img.shields.io/badge/GitHub-Profile-blue?logo=github)](https://github.com/abdul99088)

Standard LLM document uploads often behave like unpredictable black boxes—struggling with strict token limitations, dropping granular details over large texts, or introducing unwanted hallucinations. 

This repository implements a **Full-Stack Hybrid RAG (Retrieval-Augmented Generation) Architecture** that solves these limitations. By parsing incoming text into semantic vectors and pairing them with lexical keywords, this application guarantees exact, context-aware information retrieval with near-zero inference latency.

---

## 🏗️ System Architecture & Core Engineering

The platform is engineered using a decoupled client-server architecture:
### 🔍 Key Features

* **Intelligent File Parsing:** Intercepts incoming binary streams (`.pdf`, `.md`, `.txt`) and dynamically routes processing logic. Leverages `pypdf` to programmatically crawl page geometries and extract clean plain text streams.
* **Hybrid Retrieval Strategy (Dense + Sparse):** Evaluates user queries through a multi-layered verification system. It queries a **ChromaDB Vector Store** to assemble semantic candidates, then feeds those tokens into a **BM25Okapi Reranking Algorithm** to isolate exact keyword intersections.
* **Blazing-Fast Inference Pipeline:** Packages the synthesized context blocks cleanly into optimized prompt boundaries and forwards them to the `llama-3.1-8b-instant` model hosted on **Groq**, delivering low-latency responses.
* **Resilient Caching Layer:** Incorporates a local file-system caching strategy (`/tmp/rag_fallback_cache.txt`) ensuring complete system persistence and high reliability under multi-tenant cross-port container requests.

---

## 🛠️ Tech Stack & Dependencies

* **Frontend Dashboard:** Streamlit
* **Backend API Engine:** FastAPI / Uvicorn Server
* **Vector Store & Indexing:** ChromaDB (`all-MiniLM-L6-v2`)
* **Lexical Ranking:** Rank-BM25
* **LLM Engine:** Groq API SDK (`Llama-3.1-8b-instant`)
* **File Utilities:** PyPDF, IO Streams

---

## 🚀 Quickstart & Installation

### 1. Clone the Repository
```bash
git clone [https://github.com/abdul99088/Advanced-RAG-Bot.git](https://github.com/abdul99088/Advanced-RAG-Bot.git)
cd Advanced-RAG-Bot
