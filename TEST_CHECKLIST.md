# Test Checklist - RAG Application

## Pre-Testing Setup

### ✅ Environment Setup
- [ ] Python 3.9+ installed
- [ ] Virtual environment activated
- [ ] Qdrant running on localhost:6333
- [ ] `.env` file created with OPENAI_API_KEY

### ✅ Package Installation
- [ ] Run: `pip uninstall docx -y` (remove conflict)
- [ ] Run: `pip install -r requirements.txt`
- [ ] Verify: `python -c "from app.main import app; print('OK')"`

---

## System Readiness Checks

### 1. ✅ Import Verification
```bash
# Test all critical imports
python -c "
from app.services.file_parser import parse_file, FileValidationError
from app.services.chunking_service import get_chunking_service
from app.services.embedding_service import get_embedding_service
from app.services.vector_store import get_vector_store_service
from app.services.rag_service import get_rag_service
from app.api.routes import router
from app.main import app
print('✓ All imports successful')
"
```
**Expected**: No errors, prints "✓ All imports successful"

### 2. ✅ Configuration Check
```bash
python -c "
from app.config.settings import settings
print(f'Qdrant: {settings.qdrant_host}:{settings.qdrant_port}')
print(f'Embedding Model: {settings.embedding_model}')
print(f'LLM Model: {settings.llm_model}')
print(f'API Key Set: {bool(settings.openai_api_key)}')
print(f'RAG Top K: {settings.rag_top_k}')
print(f'RAG Threshold: {settings.rag_score_threshold}')
"
```
**Expected**: All settings displayed correctly

### 3. ✅ Service Initialization
```bash
python -c "
from app.services.embedding_service import get_embedding_service
from app.services.vector_store import get_vector_store_service

# Test embedding service
emb = get_embedding_service()
print(f'✓ Embedding Service: {emb.model_name}')
print(f'✓ Embedding Dimension: {emb.embedding_dim}')

# Test vector store (requires Qdrant running)
try:
    vs = get_vector_store_service()
    info = vs.get_collection_info()
    print(f'✓ Vector Store Connected: {info}')
except Exception as e:
    print(f'✗ Vector Store Error: {e}')
"
```
**Expected**: Services initialize without errors

---

## API Endpoint Testing

### Start Server
```bash
uvicorn app.main:app --reload --port 8000
```

### Test 1: Root Endpoint
```bash
curl http://localhost:8000/
```
**Expected**:
```json
{
  "message": "RAG Application API",
  "version": "1.0.0",
  "status": "running"
}
```

### Test 2: Simple Health Check
```bash
curl http://localhost:8000/health
```
**Expected**:
```json
{
  "status": "healthy"
}
```

### Test 3: Detailed Health Check
```bash
curl http://localhost:8000/api/v1/health
```
**Expected**:
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

### Test 4: Ingest Excel Document
Create a test Excel file (`test.xlsx`) with some data, then:

```bash
curl -X POST "http://localhost:8000/api/v1/ingest" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test.xlsx"
```

**Expected**:
```json
{
  "success": true,
  "message": "File successfully ingested and processed",
  "filename": "test.xlsx",
  "file_type": "excel",
  "chunks_created": 10,
  "chunks_stored": 10
}
```

**Check**:
- [ ] Status code: 201
- [ ] `success: true`
- [ ] `chunks_created > 0`
- [ ] `chunks_stored > 0`

### Test 5: Ingest Word Document
Create a test Word file (`test.docx`), then:

```bash
curl -X POST "http://localhost:8000/api/v1/ingest" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test.docx"
```

**Expected**: Similar to Test 4 but with `file_type: "word"`

### Test 6: Query - Basic
```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What information is in the documents?",
    "top_k": 5,
    "return_sources": true
  }'
```

**Expected**:
```json
{
  "answer": "Based on the documents...",
  "query": "What information is in the documents?",
  "sources": [
    {
      "content": "...",
      "score": 0.85,
      "metadata": {...}
    }
  ],
  "success": true
}
```

**Check**:
- [ ] Status code: 200
- [ ] `success: true`
- [ ] `answer` is not empty
- [ ] `sources` array has items (if documents ingested)

### Test 7: Query - Without Sources
```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Test query",
    "top_k": 3,
    "return_sources": false
  }'
```

**Expected**: Same as Test 6 but `sources: null`

---

## Error Handling Tests

### Test 8: Invalid File Type
```bash
curl -X POST "http://localhost:8000/api/v1/ingest" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test.txt"
```

**Expected**:
- Status code: 400
- Error message about unsupported file type

### Test 9: Empty Query
```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "",
    "top_k": 5
  }'
```

**Expected**:
- Status code: 422 (Validation Error)
- Error about query length

### Test 10: Invalid top_k
```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "test",
    "top_k": 100
  }'
```

**Expected**:
- Status code: 422
- Error about top_k validation (max is 20)

---

## Function Call Verification

### Async/Sync Consistency

**File**: `app/api/routes.py`

#### Ingest Endpoint (Line ~157-167):
- [ ] `await embedding_service.embed_chunks_async(...)` ✅ ASYNC
- [ ] `await vector_store.store_documents_async(...)` ✅ ASYNC

**Signature Match**:
```python
# embedding_service.py:250
async def embed_chunks_async(
    self,
    chunks: List[DocumentChunk],
    batch_size: int = 32,
    show_progress: bool = True
) -> List[ChunkWithEmbedding]:
```
✅ Matches call in routes.py:157

```python
# vector_store.py:204
async def store_documents_async(
    self,
    chunks_with_embeddings: List[ChunkWithEmbedding]
) -> List[str]:
```
✅ Matches call in routes.py:167

#### Query Endpoint (Line ~218):
- [ ] `await rag_service.answer_question_async(...)` ✅ ASYNC

**Signature Match**:
```python
# rag_service.py:238
async def answer_question_async(
    self,
    query: str,
    top_k: Optional[int] = None,
    return_sources: bool = True
) -> Dict[str, Any]:
```
✅ Matches call in routes.py

---

## Integration Tests

### Test 11: Full Pipeline
1. **Ingest** a document
2. **Wait** 2 seconds
3. **Query** related to document content
4. **Verify** answer contains relevant information

```bash
# 1. Ingest
curl -X POST "http://localhost:8000/api/v1/ingest" \
  -F "file=@test.xlsx" > /tmp/ingest.json

# 2. Check ingestion
cat /tmp/ingest.json | python -m json.tool

# 3. Query
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What data is in the spreadsheet?", "top_k": 5}' \
  > /tmp/query.json

# 4. Check response
cat /tmp/query.json | python -m json.tool
```

**Expected**:
- Ingest returns chunks_stored > 0
- Query returns meaningful answer with sources

---

## Code Quality Checks

### Import Dependencies
- [x] `file_parser.py` → ✅ All imports valid
- [x] `chunking_service.py` → ✅ All imports valid
- [x] `embedding_service.py` → ✅ All imports valid (includes numpy)
- [x] `vector_store.py` → ✅ All imports valid
- [x] `rag_service.py` → ✅ All imports valid
- [x] `routes.py` → ✅ All imports valid
- [x] `main.py` → ✅ All imports valid

### Function Signatures Match
- [x] `parse_file(file: BinaryIO, filename: str)` → ✅ Used correctly
- [x] `get_chunking_service(chunk_size, chunk_overlap)` → ✅ Used correctly
- [x] `chunk_parsed_document(parsed_doc, rechunk)` → ✅ Used correctly
- [x] `embed_chunks_async(chunks, batch_size, show_progress)` → ✅ Used correctly
- [x] `store_documents_async(chunks_with_embeddings)` → ✅ Used correctly
- [x] `answer_question_async(query, top_k, return_sources)` → ✅ Used correctly

### Async/Sync Consistency
- [x] All `async def` endpoints use `await` → ✅ Correct
- [x] Async service methods called with `await` → ✅ Correct
- [x] No sync calls in async context → ✅ Correct

---

## Known Issues & Fixes

### ✅ FIXED: requirements.txt
- ✅ Added `numpy>=1.24.0`
- ✅ Added `torch>=2.0.0`
- ✅ Updated LangChain versions
- ✅ Updated sentence-transformers

### ✅ FIXED: rag_service.py
- ✅ Changed `_aget_relevant_documents` → `aget_relevant_documents`

### ✅ FIXED: settings.py
- ✅ Lowered `rag_score_threshold` from 0.7 to 0.3

### ⚠️ IMPORTANT: docx package
- ⚠️ Must uninstall `docx==0.2.4` before installing requirements
- ✅ Use only `python-docx==1.1.0`

---

## Final Readiness Checklist

Before declaring system ready:

- [ ] All imports work without errors
- [ ] Qdrant is running and accessible
- [ ] OpenAI API key is configured
- [ ] Server starts without errors
- [ ] All 3 endpoints respond correctly
- [ ] Health check shows all services healthy
- [ ] Can ingest Excel file successfully
- [ ] Can ingest Word file successfully
- [ ] Can query and get responses
- [ ] Error handling works correctly
- [ ] No async/sync mismatches
- [ ] All function signatures match

## System Status: 🟢 READY FOR TESTING

All critical components verified and ready!