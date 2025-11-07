# RAG Application with FastAPI and Qdrant

A professional RAG (Retrieval-Augmented Generation) application built with FastAPI and Qdrant vector database.

## Features

- FastAPI REST API with CORS support
- Qdrant vector database for efficient similarity search
- Document processing for multiple formats (TXT, PDF, DOCX, CSV, XLSX)
- Sentence-transformers for text embeddings
- Docker Compose setup for easy deployment
- Environment-based configuration
- Structured service layer architecture

## Project Structure

```
rag_parse/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI application
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py          # Configuration management
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py           # Pydantic models
│   └── services/
│       ├── __init__.py
│       ├── embeddings.py        # Embedding service
│       ├── vector_store.py      # Qdrant integration
│       └── document_processor.py # Document processing
├── data/
│   └── sample_document.txt      # Sample data
├── docker-compose.yml           # Docker services
├── Dockerfile                   # FastAPI container
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variables template
└── README.md
```

## Prerequisites

- Python 3.11+
- Docker and Docker Compose
- Git

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd rag_parse
```

### 2. Create Environment File

```bash
cp .env.example .env
```

Edit `.env` file if you need to customize any settings.

### 3. Run with Docker Compose

```bash
docker-compose up -d
```

This will start:
- FastAPI application on `http://localhost:8000`
- Qdrant vector database on `http://localhost:6333`

### 4. Alternative: Local Development

If you prefer to run locally without Docker:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start Qdrant with Docker
docker-compose up -d qdrant

# Run FastAPI
uvicorn app.main:app --reload
```

## API Endpoints

### Health Check

```bash
GET http://localhost:8000/
GET http://localhost:8000/health
```

### Future Endpoints (to be implemented)

- `POST /api/v1/documents/upload` - Upload and process documents
- `POST /api/v1/query` - Search for similar documents
- `GET /api/v1/documents` - List documents
- `DELETE /api/v1/documents/{id}` - Delete a document

## Usage Example

### Python Client Example

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())
```

### Using the Services Directly

```python
from app.services.vector_store import get_vector_store_service
from app.services.document_processor import DocumentProcessor

# Process a document
processor = DocumentProcessor()
documents = processor.process_file("data/sample_document.txt")

# Add to vector store
vector_store = get_vector_store_service()
for doc in documents:
    vector_store.add_document(
        content=doc["content"],
        metadata=doc["metadata"]
    )

# Search
results = vector_store.search("What is RAG?", top_k=5)
for result in results:
    print(f"Score: {result['score']:.4f}")
    print(f"Content: {result['content'][:100]}...")
    print()
```

## Configuration

All configuration is managed through environment variables. See `.env.example` for available options:

- `QDRANT_HOST`: Qdrant server host
- `QDRANT_PORT`: Qdrant server port
- `QDRANT_COLLECTION_NAME`: Vector collection name
- `EMBEDDING_MODEL`: Sentence-transformers model name
- `CORS_ORIGINS`: Allowed CORS origins

## Development

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black app/
isort app/
```

### Type Checking

```bash
mypy app/
```

## Accessing Services

- **FastAPI Docs**: http://localhost:8000/docs
- **Qdrant Dashboard**: http://localhost:6333/dashboard
- **API**: http://localhost:8000

## Troubleshooting

### Qdrant Connection Issues

If you get connection errors to Qdrant:
1. Ensure Qdrant is running: `docker-compose ps`
2. Check logs: `docker-compose logs qdrant`
3. Verify environment variables in `.env`

### Port Already in Use

If port 8000 or 6333 is already in use:
1. Stop the conflicting service
2. Or change ports in `docker-compose.yml`

## Next Steps

To extend this application:

1. **Add API Endpoints**: Create routers in `app/api/` for document upload and querying
2. **Implement RAG Pipeline**: Integrate with LangChain for full RAG functionality
3. **Add Authentication**: Implement JWT-based authentication
4. **Add Tests**: Create unit and integration tests
5. **Add Monitoring**: Integrate logging and monitoring tools
6. **Deploy**: Set up CI/CD pipeline for deployment

## Technologies Used

- **FastAPI**: Modern web framework for building APIs
- **Qdrant**: Vector database for similarity search
- **Sentence-Transformers**: Text embedding models
- **LangChain**: Framework for LLM applications
- **Pandas**: Data processing
- **python-docx**: DOCX file processing
- **pypdf**: PDF file processing
- **Docker**: Containerization

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
