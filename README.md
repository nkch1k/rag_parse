# RAG Application - Backend Implementation

A production-ready Retrieval-Augmented Generation (RAG) system built with FastAPI, capable of processing and querying structured (Excel) and unstructured (Word) documents.

## Overview

This application demonstrates a complete RAG pipeline that:
1. **Ingests** documents (Excel .xlsx and Word .docx files)
2. **Processes** content into optimized chunks
3. **Embeds** text using sentence-transformers
4. **Stores** vectors in Qdrant database
5. **Retrieves** relevant context based on queries
6. **Generates** intelligent answers using OpenAI LLM

## Technology Stack

- **Framework**: FastAPI (Python 3.9+)
- **Vector Database**: Qdrant
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **LLM**: OpenAI GPT-3.5-turbo
- **Document Processing**: pandas, python-docx, openpyxl
- **Text Splitting**: LangChain RecursiveCharacterTextSplitter

## Project Structure

```
rag_parse/
├── app/
│   ├── main.py                    # FastAPI application entry point
│   ├── config/
│   │   └── settings.py            # Configuration management
│   ├── models/
│   │   ├── document.py            # Document data models
│   │   └── schemas.py             # API request/response schemas
│   ├── services/
│   │   ├── file_parser.py         # Excel/Word file parsing
│   │   ├── chunking_service.py    # Text chunking logic
│   │   ├── embedding_service.py   # Vector embeddings generation
│   │   ├── vector_store.py        # Qdrant vector database interface
│   │   ├── llm_service.py         # OpenAI LLM integration
│   │   └── rag_service.py         # RAG pipeline orchestration
│   └── api/
│       └── routes.py              # API endpoint definitions
├── data/
│   ├── example_sales.xlsx         # Sample Excel file
│   ├── example_products.xlsx      # Sample Excel file
│   ├── example_report.docx        # Sample Word document
│   └── example_instructions.docx  # Sample Word document
├── docker-compose.yml             # Docker services configuration
├── Dockerfile                     # Application container
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment variables template
├── load_examples.py               # Script to load example documents
└── README.md                      # This file
```

## Installation Instructions

### Prerequisites

- Python 3.9 or higher
- Docker and Docker Compose
- OpenAI API key

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd rag_parse
```

### Step 2: Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=sk-your-actual-api-key-here
```

### Step 3: Start Qdrant Vector Database

```bash
# Using Docker Compose (recommended)
docker-compose up -d qdrant

# Or standalone Docker
docker run -d -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```

### Step 4: Install Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Unix/MacOS:
source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

### Step 5: Run Application

```bash
# Development mode with auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: `http://localhost:8000`

### Step 6: Load Example Documents (Optional but Recommended)

To test the system with example data, load the provided sample files:

**Option A: Using the automated script (recommended)**

```bash
# In a new terminal (keep the API server running)
python load_examples.py
```

This script will automatically upload all 4 example files from the `data/` directory.

**Option B: Manual upload via cURL**

```bash
# Upload Excel files
curl -X POST "http://localhost:8000/api/v1/ingest" \
  -F "file=@data/example_sales.xlsx"

curl -X POST "http://localhost:8000/api/v1/ingest" \
  -F "file=@data/example_products.xlsx"

# Upload Word documents
curl -X POST "http://localhost:8000/api/v1/ingest" \
  -F "file=@data/example_report.docx"

curl -X POST "http://localhost:8000/api/v1/ingest" \
  -F "file=@data/example_instructions.docx"
```

**Option C: Using Swagger UI**

1. Open `http://localhost:8000/docs`
2. Navigate to `/api/v1/ingest` endpoint
3. Click "Try it out"
4. Upload each file from `data/` directory

**Verify upload:**

```bash
curl http://localhost:8000/api/v1/health
```

You should see `vector_store_points` showing 188 chunks (approximate total from all 4 files).

## API Endpoints

### Health Check

**GET** `/health`

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy"
}
```

### Detailed Health Check

**GET** `/api/v1/health`

```bash
curl http://localhost:8000/api/v1/health
```

**Response:**
```json
{
  "status": "healthy",
  "services": {
    "vector_store": "healthy",
    "vector_store_points": "0",
    "embedding_service": "healthy",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "embedding_dimension": "384",
    "llm_service": "configured",
    "llm_model": "gpt-3.5-turbo"
  },
  "version": "1.0.0"
}
```

### Document Ingestion

**POST** `/api/v1/ingest`

Upload and process Excel or Word documents.

**Example - Upload Excel file:**

```bash
curl -X POST "http://localhost:8000/api/v1/ingest" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data/example_sales.xlsx"
```

**Expected Response:**
```json
{
  "success": true,
  "message": "File successfully ingested and processed",
  "filename": "example_sales.xlsx",
  "file_type": "excel",
  "chunks_created": 15,
  "chunks_stored": 15
}
```

**Example - Upload Word document:**

```bash
curl -X POST "http://localhost:8000/api/v1/ingest" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data/example_report.docx"
```

**Expected Response:**
```json
{
  "success": true,
  "message": "File successfully ingested and processed",
  "filename": "example_report.docx",
  "file_type": "word",
  "chunks_created": 8,
  "chunks_stored": 8
}
```

### Query Documents

**POST** `/api/v1/query`

Ask questions and receive AI-generated answers based on ingested documents.

**Request Body:**
```json
{
  "query": "What are the total sales figures?",
  "top_k": 5,
  "return_sources": true
}
```

**Example Request:**

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the total sales figures?",
    "top_k": 5,
    "return_sources": true
  }'
```

**Expected Response:**
```json
{
  "answer": "Based on the sales data, the total sales figures are $1,234,567 across all regions. The breakdown shows North region with $456,789, South region with $345,678, East region with $234,567, and West region with $197,533.",
  "query": "What are the total sales figures?",
  "sources": [
    {
      "content": "Sales Summary\nRegion: North, Total: $456,789...",
      "score": 0.87,
      "metadata": {
        "filename": "example_sales.xlsx",
        "file_type": "excel",
        "sheet_name": "Sales_Data",
        "chunk_index": 0
      }
    },
    {
      "content": "Financial Overview\nTotal Revenue: $1,234,567...",
      "score": 0.82,
      "metadata": {
        "filename": "example_sales.xlsx",
        "file_type": "excel",
        "sheet_name": "Summary",
        "chunk_index": 1
      }
    }
  ],
  "success": true
}
```

**Example - Query without sources:**

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Summarize the company report",
    "top_k": 3,
    "return_sources": false
  }'
```

**Expected Response:**
```json
{
  "answer": "The company report highlights strong quarterly performance with revenue growth of 15% year-over-year. Key achievements include successful product launches, expansion into new markets, and improved operational efficiency.",
  "query": "Summarize the company report",
  "sources": null,
  "success": true
}
```

## Architecture and Technical Decisions

### RAG Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Document Ingestion                      │
│  1. File Upload (Excel/Word) → 2. Parse → 3. Validate       │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                       Text Chunking                          │
│  4. Split into chunks (500 chars, 50 overlap)               │
│  5. Preserve metadata (filename, sheet, paragraph)          │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                       Embedding                              │
│  6. Generate vectors using sentence-transformers            │
│  7. Batch processing (32 chunks/batch)                      │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                    Vector Storage                            │
│  8. Store in Qdrant with metadata                           │
│  9. Create indexes for fast retrieval                       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                         Query Flow                           │
│  1. User Query → 2. Embed Query → 3. Vector Search          │
│  4. Retrieve Top-K → 5. LLM Generation → 6. Response        │
└─────────────────────────────────────────────────────────────┘
```

### Key Technical Decisions

#### 1. Document Processing Strategy

**Excel Files:**
- Parse all sheets independently
- Convert tables to text format preserving structure
- Include column headers for context
- Store sheet names in metadata

**Word Files:**
- Extract text from paragraphs and tables
- Preserve document structure
- Maintain formatting context

**Rationale:** Different document types require specialized parsing to preserve semantic meaning.

#### 2. Chunking Strategy

- **Chunk Size:** 500 characters
- **Overlap:** 50 characters
- **Splitter:** LangChain RecursiveCharacterTextSplitter

**Rationale:**
- 500 chars provides enough context without exceeding embedding model limits
- 50 char overlap prevents context loss at boundaries
- Recursive splitting respects natural text boundaries (paragraphs, sentences)

#### 3. Embedding Model

**Model:** `sentence-transformers/all-MiniLM-L6-v2`
- Dimension: 384
- Fast inference
- Good balance of quality and performance

**Rationale:** Lightweight model with excellent semantic understanding, suitable for production deployment.

#### 4. Vector Database

**Qdrant** chosen for:
- Native Python support
- Efficient similarity search
- Rich metadata filtering
- Easy Docker deployment

#### 5. LLM Integration

**OpenAI GPT-3.5-turbo** with:
- Temperature: 0.1 (focused, deterministic responses)
- Max tokens: 500 (concise answers)

**Rationale:** Cost-effective model with strong reasoning capabilities for answering questions based on retrieved context.

#### 6. Asynchronous Design

All I/O operations use `async/await`:
- File processing
- Embedding generation
- Vector storage
- LLM calls

**Rationale:** Enables handling multiple concurrent requests efficiently.

## Usage Examples

### Quick Start with Example Data

The repository includes 4 example files in the `data/` directory:
- **example_sales.xlsx**: Sales transactions data (100 rows)
- **example_products.xlsx**: Product inventory (20 products)
- **example_report.docx**: Company report (unstructured text)
- **example_instructions.docx**: RAG instructions document

**Load all examples with one command:**

```bash
python load_examples.py
```

Expected output:
```
✓ Loaded 4/4 files
✓ Total chunks indexed: 188
✓ Vector store now contains: 188 points
```

### Complete Workflow

```bash
# 1. Check system health
curl http://localhost:8000/api/v1/health

# 2. Load example data (if not done yet)
python load_examples.py

# 3. Query about sales data
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the total sales figures?",
    "top_k": 5,
    "return_sources": true
  }'

# 4. Query about products
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What products are available?",
    "top_k": 3,
    "return_sources": true
  }'

# 5. Upload your own documents
curl -X POST "http://localhost:8000/api/v1/ingest" \
  -F "file=@your_document.xlsx"
```

### Interactive API Documentation

Access Swagger UI for interactive testing:
```
http://localhost:8000/docs
```

Access ReDoc for alternative documentation:
```
http://localhost:8000/redoc
```

## Configuration

All settings are managed through environment variables in `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `LLM_MODEL` | OpenAI model name | gpt-3.5-turbo |
| `LLM_TEMPERATURE` | LLM creativity (0-1) | 0.1 |
| `LLM_MAX_TOKENS` | Max response length | 500 |
| `QDRANT_HOST` | Qdrant server host | localhost |
| `QDRANT_PORT` | Qdrant server port | 6333 |
| `QDRANT_COLLECTION_NAME` | Vector collection name | documents |
| `EMBEDDING_MODEL` | Sentence transformer model | all-MiniLM-L6-v2 |
| `RAG_TOP_K` | Default retrieval count | 5 |
| `RAG_SCORE_THRESHOLD` | Min similarity score | 0.3 |

## Troubleshooting

### Qdrant Connection Error

**Problem:** `Failed to connect to Qdrant`

**Solution:**
```bash
# Check if Qdrant is running
docker ps | grep qdrant

# If not running, start it
docker-compose up -d qdrant

# Check logs
docker-compose logs qdrant
```

### OpenAI API Error

**Problem:** `OpenAI API key not found`

**Solution:** Verify `.env` file contains valid API key:
```bash
cat .env | grep OPENAI_API_KEY
```

### Import Errors

**Problem:** `ModuleNotFoundError`

**Solution:**
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Verify installation
python -c "from app.main import app; print('OK')"
```

## Testing

The application can be tested via:

1. **Swagger UI**: `http://localhost:8000/docs`
   - Interactive API testing
   - Automatic request validation
   - Response examples

2. **cURL commands**: See examples above

3. **Python client**:
```python
import requests

# Ingest document
files = {'file': open('data/example_sales.xlsx', 'rb')}
response = requests.post('http://localhost:8000/api/v1/ingest', files=files)
print(response.json())

# Query
query = {
    "query": "What are the sales figures?",
    "top_k": 5,
    "return_sources": true
}
response = requests.post('http://localhost:8000/api/v1/query', json=query)
print(response.json())
```

## License

MIT License

## Contact

For questions or issues, please contact: naumkhart@gmail.com
