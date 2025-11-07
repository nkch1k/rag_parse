from typing import List, Dict, Any, Optional, Tuple
import logging
import asyncio
from sentence_transformers import SentenceTransformer
import numpy as np
from app.models.document import DocumentChunk
from app.config.settings import settings

logger = logging.getLogger(__name__)


class ChunkWithEmbedding:
    """Container for a chunk with its embedding and metadata."""

    def __init__(
        self,
        content: str,
        embedding: List[float],
        metadata: Dict[str, Any],
        chunk_index: int
    ):
        """
        Initialize a chunk with embedding.

        Args:
            content: Text content of the chunk
            embedding: Embedding vector for the chunk
            metadata: Metadata dictionary
            chunk_index: Index of the chunk in the document
        """
        self.content = content
        self.embedding = embedding
        self.metadata = metadata
        self.chunk_index = chunk_index

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation.

        Returns:
            Dictionary with chunk data
        """
        return {
            "content": self.content,
            "embedding": self.embedding,
            "metadata": self.metadata,
            "chunk_index": self.chunk_index,
            "embedding_dimension": len(self.embedding)
        }


class EmbeddingService:
    """Service for generating text embeddings using sentence-transformers."""

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the embedding service.

        Args:
            model_name: Name of the sentence-transformers model to use
                       (default: all-MiniLM-L6-v2)
        """
        # Use provided model or default from settings
        if model_name is None:
            model_name = settings.embedding_model

        # Handle model name format
        if not model_name.startswith("sentence-transformers/"):
            model_name = f"sentence-transformers/{model_name}" if "/" not in model_name else model_name

        self.model_name = model_name
        logger.info(f"Loading embedding model: {self.model_name}")

        try:
            self.model = SentenceTransformer(self.model_name)
            embedding_dim = self.model.get_sentence_embedding_dimension()
            if embedding_dim is None:
                raise ValueError("Model returned None for embedding dimension")
            self.embedding_dim: int = embedding_dim
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            List of floats representing the embedding vector
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            # Return zero vector for empty text
            return [0.0] * self.embedding_dim

        try:
            embedding = self.model.encode(
                text,
                convert_to_tensor=False,
                normalize_embeddings=True  # Normalize for cosine similarity
            )
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    async def embed_text_async(self, text: str) -> List[float]:
        """
        Async version: Generate embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            List of floats representing the embedding vector
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_text, text)

    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts efficiently.

        Args:
            texts: List of input texts to embed
            batch_size: Number of texts to process at once
            show_progress: Whether to show progress bar

        Returns:
            List of embedding vectors
        """
        if not texts:
            logger.warning("Empty text list provided for batch embedding")
            return []

        # Filter out empty texts but keep track of indices
        valid_texts = []
        valid_indices = []
        for idx, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text)
                valid_indices.append(idx)

        if not valid_texts:
            logger.warning("No valid texts to embed")
            return [[0.0] * self.embedding_dim] * len(texts)

        try:
            logger.info(f"Generating embeddings for {len(valid_texts)} texts")
            embeddings = self.model.encode(
                valid_texts,
                batch_size=batch_size,
                convert_to_tensor=False,
                normalize_embeddings=True,  # Normalize for cosine similarity
                show_progress_bar=show_progress
            )

            # Create result list with zero vectors for empty texts
            result = [[0.0] * self.embedding_dim] * len(texts)
            for i, valid_idx in enumerate(valid_indices):
                result[valid_idx] = embeddings[i].tolist()

            return result
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            raise

    async def embed_batch_async(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> List[List[float]]:
        """
        Async version: Generate embeddings for multiple texts efficiently.

        Args:
            texts: List of input texts to embed
            batch_size: Number of texts to process at once
            show_progress: Whether to show progress bar

        Returns:
            List of embedding vectors
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.embed_batch,
            texts,
            batch_size,
            show_progress
        )

    def embed_chunks(
        self,
        chunks: List[DocumentChunk],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> List[ChunkWithEmbedding]:
        """
        Generate embeddings for a list of DocumentChunk objects.

        Args:
            chunks: List of DocumentChunk objects
            batch_size: Number of chunks to process at once
            show_progress: Whether to show progress bar

        Returns:
            List of ChunkWithEmbedding objects
        """
        if not chunks:
            logger.warning("No chunks provided for embedding")
            return []

        logger.info(f"Generating embeddings for {len(chunks)} chunks")

        # Extract text content from chunks
        texts = [chunk.content for chunk in chunks]

        # Generate embeddings in batch
        embeddings = self.embed_batch(
            texts=texts,
            batch_size=batch_size,
            show_progress=show_progress
        )

        # Create ChunkWithEmbedding objects
        chunks_with_embeddings = []
        for chunk, embedding in zip(chunks, embeddings):
            chunk_with_emb = ChunkWithEmbedding(
                content=chunk.content,
                embedding=embedding,
                metadata=chunk.metadata,
                chunk_index=chunk.chunk_index
            )
            chunks_with_embeddings.append(chunk_with_emb)

        logger.info(f"Successfully generated embeddings for {len(chunks_with_embeddings)} chunks")
        return chunks_with_embeddings

    async def embed_chunks_async(
        self,
        chunks: List[DocumentChunk],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> List[ChunkWithEmbedding]:
        """
        Async version: Generate embeddings for a list of DocumentChunk objects.

        Args:
            chunks: List of DocumentChunk objects
            batch_size: Number of chunks to process at once
            show_progress: Whether to show progress bar

        Returns:
            List of ChunkWithEmbedding objects
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.embed_chunks,
            chunks,
            batch_size,
            show_progress
        )

    def compute_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score (0 to 1)
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        # Compute cosine similarity
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return float(similarity)

    def find_similar_chunks(
        self,
        query_embedding: List[float],
        chunks_with_embeddings: List[ChunkWithEmbedding],
        top_k: int = 5
    ) -> List[Tuple[ChunkWithEmbedding, float]]:
        """
        Find the most similar chunks to a query embedding.

        Args:
            query_embedding: Query embedding vector
            chunks_with_embeddings: List of chunks with embeddings
            top_k: Number of top similar chunks to return

        Returns:
            List of tuples (chunk, similarity_score) sorted by similarity
        """
        if not chunks_with_embeddings:
            return []

        # Compute similarities
        similarities = []
        for chunk in chunks_with_embeddings:
            similarity = self.compute_similarity(query_embedding, chunk.embedding)
            similarities.append((chunk, similarity))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top k results
        return similarities[:top_k]

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.

        Returns:
            Embedding dimension size
        """
        return self.embedding_dim

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the embedding model.

        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "max_sequence_length": self.model.max_seq_length,
        }


# Global embedding service instance
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service(model_name: Optional[str] = None) -> EmbeddingService:
    """
    Get or create the global embedding service instance.

    Args:
        model_name: Optional model name to use (default: all-MiniLM-L6-v2)

    Returns:
        EmbeddingService instance
    """
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService(model_name=model_name)
    return _embedding_service