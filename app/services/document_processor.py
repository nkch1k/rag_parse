from typing import List, Dict, Any
import logging
from pathlib import Path
import pandas as pd
from docx import Document
from pypdf import PdfReader

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Service for processing various document formats."""

    @staticmethod
    def process_text_file(file_path: str) -> List[str]:
        """
        Process a plain text file.

        Args:
            file_path: Path to the text file

        Returns:
            List of text chunks
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return [content]
        except Exception as e:
            logger.error(f"Error processing text file: {e}")
            raise

    @staticmethod
    def process_pdf_file(file_path: str) -> List[str]:
        """
        Process a PDF file.

        Args:
            file_path: Path to the PDF file

        Returns:
            List of text chunks (one per page)
        """
        try:
            reader = PdfReader(file_path)
            chunks = []
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text.strip():
                    chunks.append(text)
            return chunks
        except Exception as e:
            logger.error(f"Error processing PDF file: {e}")
            raise

    @staticmethod
    def process_docx_file(file_path: str) -> List[str]:
        """
        Process a DOCX file.

        Args:
            file_path: Path to the DOCX file

        Returns:
            List of text chunks (one per paragraph)
        """
        try:
            doc = Document(file_path)
            chunks = []
            for para in doc.paragraphs:
                if para.text.strip():
                    chunks.append(para.text)
            return chunks
        except Exception as e:
            logger.error(f"Error processing DOCX file: {e}")
            raise

    @staticmethod
    def process_csv_file(file_path: str) -> List[str]:
        """
        Process a CSV file.

        Args:
            file_path: Path to the CSV file

        Returns:
            List of text chunks (one per row)
        """
        try:
            df = pd.read_csv(file_path)
            chunks = []
            for _, row in df.iterrows():
                row_text = " | ".join([f"{col}: {val}" for col, val in row.items()])
                chunks.append(row_text)
            return chunks
        except Exception as e:
            logger.error(f"Error processing CSV file: {e}")
            raise

    @staticmethod
    def process_excel_file(file_path: str) -> List[str]:
        """
        Process an Excel file.

        Args:
            file_path: Path to the Excel file

        Returns:
            List of text chunks (one per row across all sheets)
        """
        try:
            excel_file = pd.ExcelFile(file_path)
            chunks = []
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                for _, row in df.iterrows():
                    row_text = f"Sheet: {sheet_name} | " + " | ".join(
                        [f"{col}: {val}" for col, val in row.items()]
                    )
                    chunks.append(row_text)
            return chunks
        except Exception as e:
            logger.error(f"Error processing Excel file: {e}")
            raise

    @classmethod
    def process_file(cls, file_path: str) -> List[Dict[str, Any]]:
        """
        Process a file based on its extension.

        Args:
            file_path: Path to the file

        Returns:
            List of document dictionaries with content and metadata
        """
        path = Path(file_path)
        extension = path.suffix.lower()

        logger.info(f"Processing file: {path.name} (type: {extension})")

        # Process based on file type
        if extension == '.txt':
            chunks = cls.process_text_file(file_path)
        elif extension == '.pdf':
            chunks = cls.process_pdf_file(file_path)
        elif extension == '.docx':
            chunks = cls.process_docx_file(file_path)
        elif extension == '.csv':
            chunks = cls.process_csv_file(file_path)
        elif extension in ['.xlsx', '.xls']:
            chunks = cls.process_excel_file(file_path)
        else:
            raise ValueError(f"Unsupported file type: {extension}")

        # Create document objects with metadata
        documents = []
        for idx, chunk in enumerate(chunks):
            documents.append({
                "content": chunk,
                "metadata": {
                    "filename": path.name,
                    "file_type": extension,
                    "chunk_index": idx,
                    "total_chunks": len(chunks)
                }
            })

        logger.info(f"Extracted {len(documents)} chunks from {path.name}")
        return documents
