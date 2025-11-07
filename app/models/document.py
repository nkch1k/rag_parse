from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum


class FileType(str, Enum):
    EXCEL = "excel"
    WORD = "word"
    PDF = "pdf"
    TEXT = "text"


class DocumentChunk(BaseModel):
    """Schema for a document chunk with metadata."""
    content: str = Field(..., description="Text content of the chunk")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")
    chunk_index: int = Field(..., description="Index of this chunk in the document", ge=0)

    @validator('content')
    def content_not_empty(cls, v):
        """Ensure content is not empty."""
        if not v or not v.strip():
            raise ValueError("Content cannot be empty")
        return v


class ParsedDocument(BaseModel):
    """Schema for a fully parsed document."""
    filename: str = Field(..., description="Original filename")
    file_type: FileType = Field(..., description="Type of file parsed")
    chunks: List[DocumentChunk] = Field(default_factory=list, description="List of text chunks")
    total_chunks: int = Field(..., description="Total number of chunks", ge=0)
    file_size_bytes: Optional[int] = Field(None, description="File size in bytes", ge=0)
    parsed_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp when parsed")

    class Config:
        use_enum_values = True


class ExcelMetadata(BaseModel):
    """Specific metadata for Excel files."""
    sheet_name: str = Field(..., description="Name of the Excel sheet")
    row_number: int = Field(..., description="Row number in the sheet", ge=0)
    column_names: List[str] = Field(default_factory=list, description="Column names from the sheet")
    total_rows: int = Field(..., description="Total rows in the sheet", ge=0)


class WordMetadata(BaseModel):
    """Specific metadata for Word files."""
    paragraph_number: int = Field(..., description="Paragraph number in the document", ge=0)
    total_paragraphs: int = Field(..., description="Total paragraphs in the document", ge=0)
    has_formatting: bool = Field(default=False, description="Whether paragraph has special formatting")


class FileValidationError(Exception):
    """Custom exception for file validation errors."""
    pass