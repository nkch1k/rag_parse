from typing import List, Dict, Any, Optional
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
from app.models.document import DocumentChunk, ParsedDocument, FileType

logger = logging.getLogger(__name__)


class ChunkingService:
    """Service for chunking documents into smaller pieces for embedding."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None
    ):
        """
        Initialize the chunking service.

        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            separators: List of separators to use for splitting (default: ["\\n\\n", "\\n", ". ", " ", ""])
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Default separators prioritize natural text boundaries
        if separators is None:
            separators = ["\n\n", "\n", ". ", " ", ""]

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
        )

        logger.info(
            f"Initialized ChunkingService with chunk_size={chunk_size}, "
            f"chunk_overlap={chunk_overlap}"
        )

    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """
        Chunk plain text into smaller pieces.

        Args:
            text: Text content to chunk
            metadata: Optional metadata to attach to each chunk

        Returns:
            List of DocumentChunk objects
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for chunking")
            return []

        # Split text into chunks
        text_chunks = self.text_splitter.split_text(text)

        # Create DocumentChunk objects
        chunks = []
        for idx, chunk_text in enumerate(text_chunks):
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata.update({
                "chunk_method": "recursive_text_splitter",
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
            })

            chunk = DocumentChunk(
                content=chunk_text,
                metadata=chunk_metadata,
                chunk_index=idx
            )
            chunks.append(chunk)

        logger.debug(f"Created {len(chunks)} chunks from text")
        return chunks

    def chunk_excel_by_rows(
        self,
        df: pd.DataFrame,
        sheet_name: str,
        filename: str,
        rows_per_chunk: int = 10
    ) -> List[DocumentChunk]:
        """
        Chunk Excel data by rows, keeping headers as context in each chunk.

        Args:
            df: DataFrame containing Excel data
            sheet_name: Name of the Excel sheet
            filename: Original filename
            rows_per_chunk: Number of data rows to include per chunk

        Returns:
            List of DocumentChunk objects with Excel-specific metadata
        """
        if df.empty:
            logger.warning(f"Empty DataFrame for sheet: {sheet_name}")
            return []

        chunks = []
        column_names = df.columns.tolist()
        total_rows = len(df)

        # Create header context (will be included in each chunk)
        header_text = " | ".join([f"{col}" for col in column_names])

        # Process rows in groups
        for chunk_idx, start_row in enumerate(range(0, total_rows, rows_per_chunk)):
            end_row = min(start_row + rows_per_chunk, total_rows)
            chunk_df = df.iloc[start_row:end_row]

            # Build chunk content with headers as context
            row_texts = []
            for _, row in chunk_df.iterrows():
                row_parts = []
                for col_name, value in row.items():
                    if pd.notna(value):
                        row_parts.append(f"{col_name}: {value}")
                if row_parts:
                    row_texts.append(" | ".join(row_parts))

            if not row_texts:
                continue

            # Combine header context with row data
            content = f"Headers: {header_text}\n\nData:\n" + "\n".join(row_texts)

            # Create chunk with Excel-specific metadata
            chunk = DocumentChunk(
                content=content,
                metadata={
                    "sheet_name": sheet_name,
                    "filename": filename,
                    "file_type": "excel",
                    "column_names": column_names,
                    "start_row": int(start_row),
                    "end_row": int(end_row),
                    "total_rows": int(total_rows),
                    "rows_per_chunk": rows_per_chunk,
                    "chunk_method": "excel_by_rows",
                },
                chunk_index=chunk_idx
            )
            chunks.append(chunk)

        logger.info(
            f"Created {len(chunks)} chunks from Excel sheet '{sheet_name}' "
            f"({total_rows} rows, {rows_per_chunk} rows per chunk)"
        )
        return chunks

    def chunk_parsed_document(
        self,
        parsed_doc: ParsedDocument,
        rechunk: bool = False,
        rows_per_chunk: int = 10
    ) -> List[DocumentChunk]:
        """
        Chunk a parsed document based on its file type.

        Args:
            parsed_doc: ParsedDocument object to chunk
            rechunk: If True, rechunk text content; if False, use existing chunks for non-Excel
            rows_per_chunk: Number of rows per chunk for Excel files

        Returns:
            List of DocumentChunk objects
        """
        logger.info(f"Chunking document: {parsed_doc.filename} (type: {parsed_doc.file_type})")

        # For Excel files, we always rechunk by rows if requested
        if parsed_doc.file_type == FileType.EXCEL and rechunk:
            logger.warning(
                "Excel rechunking requires original DataFrame data. "
                "Using existing chunks from parsed document."
            )
            return parsed_doc.chunks

        # For non-Excel files, either rechunk or return existing chunks
        if not rechunk:
            logger.debug("Using existing chunks from parsed document")
            return parsed_doc.chunks

        # Rechunk text-based documents
        all_chunks = []
        for original_chunk in parsed_doc.chunks:
            # Rechunk the content
            new_chunks = self.chunk_text(
                text=original_chunk.content,
                metadata=original_chunk.metadata
            )

            # Update chunk indices to be sequential across all chunks
            for new_chunk in new_chunks:
                new_chunk.chunk_index = len(all_chunks)
                all_chunks.append(new_chunk)

        logger.info(
            f"Rechunked document: {parsed_doc.filename}. "
            f"Original chunks: {len(parsed_doc.chunks)}, New chunks: {len(all_chunks)}"
        )
        return all_chunks

    def update_chunk_size(self, chunk_size: int, chunk_overlap: Optional[int] = None):
        """
        Update the chunk size and optionally the overlap for the text splitter.

        Args:
            chunk_size: New chunk size in characters
            chunk_overlap: New overlap size (if None, keeps existing overlap)
        """
        self.chunk_size = chunk_size
        if chunk_overlap is not None:
            self.chunk_overlap = chunk_overlap

        # Recreate text splitter with new settings
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.text_splitter._separators,
            length_function=len,
        )

        logger.info(
            f"Updated chunk settings: chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap}"
        )


# Global chunking service instance
_chunking_service: Optional[ChunkingService] = None


def get_chunking_service(
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> ChunkingService:
    """
    Get or create the global chunking service instance.

    Args:
        chunk_size: Maximum size of each chunk in characters
        chunk_overlap: Number of characters to overlap between chunks

    Returns:
        ChunkingService instance
    """
    global _chunking_service
    if _chunking_service is None:
        _chunking_service = ChunkingService(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    return _chunking_service