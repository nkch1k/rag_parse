from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List, Optional, Dict, Any
import logging
import asyncio
import uuid
from app.config.settings import settings
from app.services.embedding_service import get_embedding_service, ChunkWithEmbedding

logger = logging.getLogger(__name__)


class VectorStoreService:
    """Service for managing vector storage with Qdrant."""

    def __init__(
        self,
        host: str = None,
        port: int = None,
        collection_name: str = None
    ):
        """
        Initialize the vector store service.

        Args:
            host: Qdrant host address
            port: Qdrant port number
            collection_name: Name of the collection to use
        """
        self.host = host or settings.qdrant_host
        self.port = port or settings.qdrant_port
        self.collection_name = collection_name or settings.qdrant_collection_name

        try:
            logger.info(f"Connecting to Qdrant at {self.host}:{self.port}")
            self.client = QdrantClient(host=self.host, port=self.port)
            self.embedding_service = get_embedding_service()

            # Initialize collection
            self.init_collection()
        except Exception as e:
            logger.error(f"Failed to initialize VectorStoreService: {e}")
            raise

    def init_collection(self):
        """Initialize Qdrant collection if it doesn't exist."""
        try:
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]

            if self.collection_name not in collection_names:
                logger.info(f"Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_service.get_embedding_dimension(),
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Collection '{self.collection_name}' created successfully")
            else:
                logger.info(f"Collection '{self.collection_name}' already exists")
        except Exception as e:
            logger.error(f"Error initializing collection: {e}")
            raise

    def _ensure_collection_exists(self):
        """Ensure collection exists, create if missing."""
        try:
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]

            if self.collection_name not in collection_names:
                logger.warning(f"Collection '{self.collection_name}' missing, recreating...")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_service.get_embedding_dimension(),
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Collection '{self.collection_name}' recreated successfully")
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {e}")
            raise

    def add_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None
    ) -> str:
        """
        Add a single document to the vector store.

        Args:
            content: Document content
            metadata: Additional metadata
            doc_id: Optional document ID (generated if not provided)

        Returns:
            Document ID
        """
        try:
            doc_id = doc_id or str(uuid.uuid4())
            embedding = self.embedding_service.embed_text(content)

            payload = {
                "content": content,
                "metadata": metadata or {}
            }

            point = PointStruct(
                id=doc_id,
                vector=embedding,
                payload=payload
            )

            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )

            logger.info(f"Added document with ID: {doc_id}")
            return doc_id
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            raise

    def add_documents_batch(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Add multiple documents to the vector store.

        Args:
            documents: List of documents with 'content' and optional 'metadata' and 'id'

        Returns:
            List of document IDs
        """
        try:
            contents = [doc["content"] for doc in documents]
            embeddings = self.embedding_service.embed_batch(contents)

            points = []
            doc_ids = []

            for doc, embedding in zip(documents, embeddings):
                doc_id = doc.get("id", str(uuid.uuid4()))
                doc_ids.append(doc_id)

                payload = {
                    "content": doc["content"],
                    "metadata": doc.get("metadata", {})
                }

                points.append(PointStruct(
                    id=doc_id,
                    vector=embedding,
                    payload=payload
                ))

            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

            logger.info(f"Added {len(doc_ids)} documents to collection")
            return doc_ids
        except Exception as e:
            logger.error(f"Error adding documents batch: {e}")
            raise

    def store_documents(
        self,
        chunks_with_embeddings: List[ChunkWithEmbedding]
    ) -> List[str]:
        """
        Store chunks with embeddings in Qdrant.

        Args:
            chunks_with_embeddings: List of ChunkWithEmbedding objects

        Returns:
            List of document IDs
        """
        try:
            # Ensure collection exists before storing
            self._ensure_collection_exists()

            if not chunks_with_embeddings:
                logger.warning("No chunks to store")
                return []

            points = []
            doc_ids = []

            for chunk in chunks_with_embeddings:
                doc_id = str(uuid.uuid4())
                doc_ids.append(doc_id)

                payload = {
                    "content": chunk.content,
                    "metadata": chunk.metadata,
                    "chunk_index": chunk.chunk_index
                }

                points.append(PointStruct(
                    id=doc_id,
                    vector=chunk.embedding,
                    payload=payload
                ))

            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

            logger.info(f"Stored {len(doc_ids)} chunks in collection")
            return doc_ids
        except Exception as e:
            logger.error(f"Error storing documents: {e}")
            raise

    async def store_documents_async(
        self,
        chunks_with_embeddings: List[ChunkWithEmbedding]
    ) -> List[str]:
        """
        Async version: Store chunks with embeddings in Qdrant.

        Args:
            chunks_with_embeddings: List of ChunkWithEmbedding objects

        Returns:
            List of document IDs
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.store_documents,
            chunks_with_embeddings
        )

    def search(
        self,
        query_text: str,
        top_k: int = 5,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents by query text.

        Args:
            query_text: Search query text
            top_k: Number of results to return (default: 5)
            score_threshold: Minimum similarity score (optional)

        Returns:
            List of search results with scores, content, and metadata
        """
        try:
            # Ensure collection exists before searching
            self._ensure_collection_exists()

            # Generate embedding for query
            query_embedding = self.embedding_service.embed_text(query_text)

            # Search in Qdrant
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                score_threshold=score_threshold
            )

            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "id": result.id,
                    "score": result.score,
                    "content": result.payload.get("content", ""),
                    "metadata": result.payload.get("metadata", {}),
                    "chunk_index": result.payload.get("chunk_index")
                })

            logger.info(f"Found {len(formatted_results)} results for query: '{query_text[:50]}...'")
            return formatted_results
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            raise

    async def search_async(
        self,
        query_text: str,
        top_k: int = 5,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Async version: Search for similar documents by query text.

        Args:
            query_text: Search query text
            top_k: Number of results to return (default: 5)
            score_threshold: Minimum similarity score (optional)

        Returns:
            List of search results with scores, content, and metadata
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.search,
            query_text,
            top_k,
            score_threshold
        )

    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document by ID.

        Args:
            doc_id: Document ID to delete

        Returns:
            True if successful
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=[doc_id]
            )
            logger.info(f"Deleted document: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {e}")
            raise

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection.

        Returns:
            Collection information
        """
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            raise


# Global vector store instance
vector_store_service = None


def get_vector_store_service() -> VectorStoreService:
    """
    Get or create the global vector store service instance.

    Returns:
        VectorStoreService instance
    """
    global vector_store_service
    if vector_store_service is None:
        vector_store_service = VectorStoreService()
    return vector_store_service
