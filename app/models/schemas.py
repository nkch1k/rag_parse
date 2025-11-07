from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class DocumentBase(BaseModel):
    """Base schema for document data."""
    content: str = Field(..., description="Document content")
    metadata: Optional[dict] = Field(default_factory=dict, description="Additional metadata")


class DocumentCreate(DocumentBase):
    """Schema for creating a new document."""
    pass


class DocumentResponse(DocumentBase):
    """Schema for document response."""
    id: str = Field(..., description="Document ID")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")

    class Config:
        from_attributes = True


class QueryRequest(BaseModel):
    """Schema for query request."""
    query: str = Field(..., description="Search query text", min_length=1)
    top_k: int = Field(default=5, description="Number of results to return", ge=1, le=20)
    score_threshold: Optional[float] = Field(
        default=0.7,
        description="Minimum similarity score threshold",
        ge=0.0,
        le=1.0
    )


class SearchResult(BaseModel):
    """Schema for individual search result."""
    id: str = Field(..., description="Document ID")
    content: str = Field(..., description="Document content")
    score: float = Field(..., description="Similarity score")
    metadata: Optional[dict] = Field(default_factory=dict, description="Document metadata")


class QueryResponse(BaseModel):
    """Schema for query response."""
    query: str = Field(..., description="Original query")
    results: List[SearchResult] = Field(default_factory=list, description="Search results")
    total_results: int = Field(..., description="Total number of results")


class FileUploadResponse(BaseModel):
    """Schema for file upload response."""
    filename: str = Field(..., description="Uploaded filename")
    document_count: int = Field(..., description="Number of documents extracted")
    message: str = Field(..., description="Status message")
