---
title: US Tax And Legal RAG
emoji: 📚
colorFrom: purple
colorTo: indigo
sdk: docker
app_port: 7860
---



# Minimal Hybrid RAG System (US Tax & Legal)

A modular, high-performance Retrieval-Augmented Generation (RAG) system built with **FastAPI**, **FAISS**, **BM25**, and **Mistral AI**. Designed specifically for tax and legal document corpora.

---

## 🚀 Features

* **Hybrid Search Retrieval**: Combines semantic vector search (`FAISS`) and exact keyword matching (`BM25`) using **Reciprocal Rank Fusion (RRF)**.
* **Mistral AI Answer Generation**: Uses the Mistral AI API for generating precise, context-grounded responses.
* **Grounded Citations**: Automatically extracts document source references and page numbers, then cross-checks them against retrieved chunks to filter out hallucinations.
* **Fast Startup & Caching**: Tracks changes to the `data/` directory via MD5 hashing to load cached indexes in **<1 second** instead of re-embedding.
* **Lock-Checked Ingestion**: Employs cross-process file-locking (`index.lock`) to prevent ingestion race conditions when running multiple API workers.
* **Interactive UI**: Serves a clean, premium glassmorphic dark-mode search client directly from the FastAPI root endpoint.

---

## 📁 Directory Structure

```
RAG_1/
├── src/
│   ├── api/
│   │   ├── app.py              # FastAPI endpoints
│   │   ├── schemas.py          # Request & response Pydantic models
│   │   └── templates/
│   │       └── index.html      # Glassmorphic search frontend
│   ├── config/
│   │   └── config.py           # Configuration variables and LLM initialization
│   ├── document_ingestion/
│   │   └── document_processor.py # PDF directory loading & splitting
│   └── vectorstore/
│       └── hybrid_retriever.py # FAISS + BM25 RRF hybrid retrieval
├── deprecated/                 # Previous LangGraph/Streamlit files
├── data/                       # Directory containing source legal PDFs
├── api_main.py                 # FastAPI runner entrypoint
├── pyproject.toml              # Project configuration and metadata
└── requirements.txt            # Dependency listings
```

---

## 🛠️ Setup & Installation

### 1. Prerequisites
Ensure you have Python 3.10+ and `uv` or `pip` installed.

### 2. Configure Environment Variables
Create a `.env` file in the root of the project:
```env
MISTRAL_API_KEY=your-mistral-api-key-here
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Or if using `uv`):*
```bash
uv pip install -r requirements.txt
```

### 4. Load Legal Documents
Place your tax and legal PDF documents inside the `data/` directory.

---

## 🏃 Run the Application

Start the API server:
```bash
python api_main.py
```
Or with `uv`:
```bash
uv run python api_main.py
```

Open your browser and navigate to:
👉 [**http://localhost:8000/**](http://localhost:8000/)

---

## 🧪 API Endpoints

* **`GET /`**: Renders the web search UI.
* **`POST /query`**: Performs RRF hybrid search and returns a structured answer with verified citations.
  ```json
  {
    "question": "what is general provisions?"
  }
  ```
* **`GET /health`**: Returns the health status and the total count of loaded document chunks.

---

## 🌐 Deployment Guide

### ⚠️ Critical Security Rules
* **NEVER commit `.env` files, API keys, or secrets** to git.
* Set all secrets (e.g. `MISTRAL_API_KEY`) via each platform's environment variables or secrets console.

---

### Option 1: Hugging Face Spaces (Recommended & Free)
Hugging Face Spaces is the ideal hosting option for this system because it natively supports Docker and running Python packages like `torch` and `sentence-transformers` locally on a free CPU basic tier (16GB RAM).

#### Step-by-Step Deployment:
1. Create a free account on [Hugging Face](https://huggingface.co/).
2. Go to **Spaces** -> **Create new Space**.
3. Choose the following settings:
   * **Space name**: e.g., `tax-legal-rag`
   * **SDK**: **Docker**
   * **Docker template**: **Blank**
   * **Space hardware**: **CPU basic (free)**
   * **Visibility**: Public or Private
4. Create the Space.
5. Add your `MISTRAL_API_KEY` environment variable:
   * Go to **Settings** in your Space dashboard.
   * Under **Variables and secrets**, click **New secret**.
   * Set name to `MISTRAL_API_KEY` and paste your Mistral API key.
6. Push the project files (including the `Dockerfile`, `src/`, `data/`, `api_main.py`, and `requirements.txt`) to the Space git repository:
   ```bash
   git remote add hf https://huggingface.co/spaces/<your-username>/<your-space-name>
   git push hf main
   ```
7. Hugging Face will automatically build and run the Docker container. Once running, your app is accessible at `https://huggingface.co/spaces/<your-username>/<your-space-name>`.

---

### Option 2: Render (Free Web Service)
Render is a cloud hosting provider that allows deploying Web Services directly from GitHub.

#### Step-by-Step Deployment:
1. Create an account on [Render](https://render.com/).
2. Click **New +** -> **Web Service**.
3. Connect your GitHub repository.
4. Set the following configuration:
   * **Runtime**: **Docker**
   * **Instance Type**: **Free**
5. Click **Advanced** and add the environment variable:
   * Key: `MISTRAL_API_KEY` | Value: `<your-mistral-api-key>`
6. Deploy the web service. Render will read the root `Dockerfile` and spin up the server.

---

### Option 3: Vercel (Static Frontend Only + Remote Backend)
Vercel is an exceptional platform for frontends, but deploying this Python RAG backend directly on Vercel Serverless is **not possible** because:
1. The `sentence-transformers` package requires PyTorch, which is over 1GB uncompressed. This exceeds Vercel's strict **250MB size limit** for serverless functions.
2. Building or loading a FAISS index statelessly on every serverless invocation causes severe timeouts and performance overhead.

#### Deployment Architecture:
1. Deploy the backend on **Hugging Face Spaces** (Option 1) or **Render** (Option 2).
2. Configure the frontend (`src/api/templates/index.html`) to query the backend URL instead of relative `/query`.
3. Deploy the frontend code to Vercel as a static site.

