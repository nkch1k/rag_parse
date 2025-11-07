# RAG Application Setup Guide

## Prerequisites

- Python 3.9+
- Docker (for Qdrant)
- OpenAI API Key

## Installation Steps

### 1. Fix package conflicts (IMPORTANT!)

```bash
# Remove conflicting package
pip uninstall docx -y

# Install requirements
pip install -r requirements.txt
```

**Note**: The package `docx` (version 0.2.4) conflicts with `python-docx`. Make sure to uninstall `docx` first!

### 2. Set up environment variables

```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=sk-your-actual-key-here
```

### 3. Start Qdrant vector database

```bash
# Using Docker
docker run -d -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```

### 4. Verify installation

```bash
# Test imports
python -c "from app.main import app; print('✓ All imports successful')"
```

### 5. Start the API server

```bash
# Development mode with auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

Once running, access:
- **API Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health
- **Detailed Health**: http://localhost:8000/api/v1/health

## Testing the API

### 1. Health Check
```bash
curl http://localhost:8000/api/v1/health
```

### 2. Upload a Document
```bash
curl -X POST "http://localhost:8000/api/v1/ingest" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_document.xlsx"
```

### 3. Query the System
```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is in the document?",
    "top_k": 5,
    "return_sources": true
  }'
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'exceptions'"
**Solution**: You have the wrong `docx` package installed. Run:
```bash
pip uninstall docx -y
pip install python-docx==1.1.0
```

### Issue: "Connection to Qdrant failed"
**Solution**: Make sure Qdrant is running:
```bash
docker ps | grep qdrant
# If not running, start it with the command in step 3
```

### Issue: "OpenAI API key not found"
**Solution**: Check your `.env` file has:
```
OPENAI_API_KEY=sk-your-actual-key-here
```

## Project Structure

```
rag_parse/
├── app/
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py          # API endpoints
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py        # Configuration
│   ├── models/
│   │   ├── __init__.py
│   │   ├── document.py        # Pydantic models
│   │   └── schemas.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── chunking_service.py
│   │   ├── embedding_service.py
│   │   ├── file_parser.py
│   │   ├── llm_service.py
│   │   ├── rag_service.py
│   │   └── vector_store.py
│   └── main.py                # FastAPI application
├── .env.example               # Environment variables template
├── requirements.txt           # Python dependencies
├── SETUP.md                   # This file
└── README.md
```

## Features

- ✅ Excel file parsing (.xlsx, .xls, .xlsm)
- ✅ Word document parsing (.docx, .doc)
- ✅ Automatic text chunking
- ✅ Semantic embeddings with sentence-transformers
- ✅ Vector storage with Qdrant
- ✅ RAG-based question answering with OpenAI
- ✅ Async/await support
- ✅ Comprehensive error handling
- ✅ API documentation with Swagger UI