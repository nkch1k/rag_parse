from fastapi import APIRouter, UploadFile, File, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
import tempfile
import os
from pathlib import Path

from app.services.file_parser import parse_file, FileValidationError
from app.services.chunking_service import get_chunking_service
from app.services.embedding_service import get_embedding_service
from app.services.vector_store import get_vector_store_service
from app.services.rag_service import get_rag_service

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


# ============================================================================
# Request/Response Models
# ============================================================================

class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    query: str = Field(..., min_length=1, max_length=1000, description="Question to ask")
    top_k: Optional[int] = Field(5, ge=1, le=20, description="Number of documents to retrieve")
    return_sources: bool = Field(True, description="Whether to return source documents")


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    answer: str = Field(..., description="Generated answer")
    query: str = Field(..., description="Original query")
    sources: Optional[List[Dict[str, Any]]] = Field(None, description="Source documents used")
    success: bool = Field(True, description="Whether the query was successful")


class IngestResponse(BaseModel):
    """Response model for ingest endpoint."""
    success: bool = Field(..., description="Whether ingestion was successful")
    message: str = Field(..., description="Status message")
    filename: str = Field(..., description="Name of ingested file")
    file_type: str = Field(..., description="Type of file ingested")
    chunks_created: int = Field(..., description="Number of chunks created")
    chunks_stored: int = Field(..., description="Number of chunks stored in vector DB")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Health status")
    services: Dict[str, str] = Field(..., description="Status of individual services")
    version: str = Field(..., description="API version")


class ErrorResponse(BaseModel):
    """Response model for errors."""
    success: bool = Field(False, description="Always false for errors")
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")


# ============================================================================
# Endpoints
# ============================================================================

@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and process files",
    description="Upload Excel (.xlsx, .xls) or Word (.docx) files for ingestion into the RAG system",
    responses={
        201: {"description": "File successfully ingested"},
        400: {"model": ErrorResponse, "description": "Invalid file or validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error during processing"}
    }
)
async def ingest_document(
    file: UploadFile = File(..., description="Excel or Word file to ingest")
) -> IngestResponse:
    """
    Upload and process a document file.

    This endpoint:
    1. Validates the uploaded file (type and size)
    2. Parses the file content
    3. Chunks the content into smaller pieces
    4. Generates embeddings for each chunk
    5. Stores embeddings in the vector database

    Supported file types:
    - Excel: .xlsx, .xls, .xlsm
    - Word: .docx, .doc
    """
    temp_file_path = None

    try:
        logger.info(f"Starting ingestion for file: {file.filename}")

        # Validate filename
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No filename provided"
            )

        # Check file extension
        file_ext = Path(file.filename).suffix.lower()
        supported_extensions = {'.xlsx', '.xls', '.xlsm', '.docx', '.doc'}

        if file_ext not in supported_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type: {file_ext}. Supported types: {', '.join(supported_extensions)}"
            )

        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            temp_file_path = temp_file.name
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()

        logger.debug(f"Saved uploaded file to temporary location: {temp_file_path}")

        # Parse the file
        try:
            with open(temp_file_path, 'rb') as f:
                parsed_doc = parse_file(f, file.filename)
        except FileValidationError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )

        logger.info(f"Parsed document: {parsed_doc.total_chunks} initial chunks")

        # Get chunking service and rechunk if needed
        chunking_service = get_chunking_service(chunk_size=500, chunk_overlap=50)

        # For text-based documents, rechunk; for Excel, use existing chunks
        if parsed_doc.file_type == "excel":
            chunks = parsed_doc.chunks
            logger.info("Using existing chunks for Excel file")
        else:
            chunks = chunking_service.chunk_parsed_document(
                parsed_doc=parsed_doc,
                rechunk=True
            )
            logger.info(f"Rechunked document: {len(chunks)} final chunks")

        # Generate embeddings
        embedding_service = get_embedding_service()
        chunks_with_embeddings = await embedding_service.embed_chunks_async(
            chunks=chunks,
            batch_size=32,
            show_progress=False
        )

        logger.info(f"Generated embeddings for {len(chunks_with_embeddings)} chunks")

        # Store in vector database
        vector_store = get_vector_store_service()
        doc_ids = await vector_store.store_documents_async(chunks_with_embeddings)

        logger.info(f"Stored {len(doc_ids)} chunks in vector database")

        return IngestResponse(
            success=True,
            message="File successfully ingested and processed",
            filename=file.filename,
            file_type=parsed_doc.file_type,
            chunks_created=len(chunks),
            chunks_stored=len(doc_ids)
        )

    except HTTPException:
        raise
    except FileValidationError as e:
        logger.error(f"File validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process file: {str(e)}"
        )
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logger.debug(f"Cleaned up temporary file: {temp_file_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file: {e}")


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask questions and get answers",
    description="Submit a question and receive an AI-generated answer based on ingested documents",
    responses={
        200: {"description": "Successfully generated answer"},
        400: {"model": ErrorResponse, "description": "Invalid query"},
        500: {"model": ErrorResponse, "description": "Internal server error during query processing"}
    }
)
async def query_documents(request: QueryRequest) -> QueryResponse:
    """
    Ask a question and get an AI-generated answer based on ingested documents.

    This endpoint:
    1. Receives a user question
    2. Searches for relevant document chunks using semantic similarity
    3. Generates an answer using the LLM with retrieved context
    4. Optionally returns source documents used for the answer

    The RAG (Retrieval-Augmented Generation) process ensures answers are grounded
    in your ingested documents.
    """
    try:
        logger.info(f"Processing query: {request.query[:100]}...")

        # Get RAG service
        rag_service = get_rag_service()

        # Process query
        result = await rag_service.answer_question_async(
            query=request.query,
            top_k=request.top_k,
            return_sources=request.return_sources
        )

        logger.info("Successfully generated answer")

        return QueryResponse(
            answer=result["answer"],
            query=result["query"],
            sources=result.get("sources") if request.return_sources else None,
            success=True
        )

    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process query: {str(e)}"
        )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check the health status of the API and its services",
    responses={
        200: {"description": "Service is healthy"},
        503: {"model": ErrorResponse, "description": "Service is unhealthy"}
    }
)
async def health_check() -> HealthResponse:
    """
    Perform a health check on the API and its dependencies.

    This endpoint checks:
    - Vector database connectivity (Qdrant)
    - Embedding service status
    - LLM service configuration

    Returns 200 if all services are operational, 503 otherwise.
    """
    services_status = {}
    overall_healthy = True

    try:
        # Check vector store
        try:
            vector_store = get_vector_store_service()
            collection_info = vector_store.get_collection_info()
            services_status["vector_store"] = "healthy"
            services_status["vector_store_points"] = str(collection_info.get("points_count", 0))
        except Exception as e:
            logger.error(f"Vector store health check failed: {e}")
            services_status["vector_store"] = f"unhealthy: {str(e)}"
            overall_healthy = False

        # Check embedding service
        try:
            embedding_service = get_embedding_service()
            model_info = embedding_service.get_model_info()
            services_status["embedding_service"] = "healthy"
            services_status["embedding_model"] = model_info["model_name"]
            services_status["embedding_dimension"] = str(model_info["embedding_dimension"])
        except Exception as e:
            logger.error(f"Embedding service health check failed: {e}")
            services_status["embedding_service"] = f"unhealthy: {str(e)}"
            overall_healthy = False

        # Check LLM service
        try:
            from app.config.settings import settings
            if settings.openai_api_key:
                services_status["llm_service"] = "configured"
                services_status["llm_model"] = settings.llm_model
            else:
                services_status["llm_service"] = "not_configured"
                overall_healthy = False
        except Exception as e:
            logger.error(f"LLM service health check failed: {e}")
            services_status["llm_service"] = f"unhealthy: {str(e)}"
            overall_healthy = False

        # Determine overall status
        if not overall_healthy:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="One or more services are unhealthy"
            )

        from app.config.settings import settings
        return HealthResponse(
            status="healthy",
            services=services_status,
            version=settings.api_version
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check failed: {str(e)}"
        )
