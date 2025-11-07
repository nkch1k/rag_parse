from typing import List, Dict, Any, Iterator, Optional
import logging
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.schema.retriever import BaseRetriever
from langchain.chains import RetrievalQA
from langchain.callbacks.base import BaseCallbackHandler
from app.config.settings import settings
from app.services.vector_store import get_vector_store_service
from app.services.llm_service import get_llm_service

logger = logging.getLogger(__name__)


class VectorStoreRetriever(BaseRetriever):
    """Custom retriever that wraps our VectorStoreService."""

    def __init__(self, vector_store_service, top_k: int = 5, score_threshold: Optional[float] = None):
        """
        Initialize retriever.

        Args:
            vector_store_service: VectorStoreService instance
            top_k: Number of documents to retrieve
            score_threshold: Minimum similarity score
        """
        super().__init__()
        self.vector_store = vector_store_service
        self.top_k = top_k
        self.score_threshold = score_threshold

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        Get relevant documents for a query.

        Args:
            query: Search query

        Returns:
            List of LangChain Document objects
        """
        try:
            # Search using our vector store
            results = self.vector_store.search(
                query_text=query,
                top_k=self.top_k,
                score_threshold=self.score_threshold
            )

            # Convert to LangChain Document format
            documents = []
            for result in results:
                doc = Document(
                    page_content=result["content"],
                    metadata={
                        "score": result["score"],
                        "chunk_index": result.get("chunk_index"),
                        **result.get("metadata", {})
                    }
                )
                documents.append(doc)

            return documents
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            raise

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """
        Async version of document retrieval using native async methods.

        Args:
            query: Search query

        Returns:
            List of LangChain Document objects
        """
        try:
            # Use async search method
            results = await self.vector_store.search_async(
                query_text=query,
                top_k=self.top_k,
                score_threshold=self.score_threshold
            )

            # Convert to LangChain Document format
            documents = []
            for result in results:
                doc = Document(
                    page_content=result["content"],
                    metadata={
                        "score": result["score"],
                        "chunk_index": result.get("chunk_index"),
                        **result.get("metadata", {})
                    }
                )
                documents.append(doc)

            return documents
        except Exception as e:
            logger.error(f"Error retrieving documents async: {e}")
            raise


class StreamingCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming responses."""

    def __init__(self):
        self.tokens = []
        self.streaming = True

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Called when LLM generates a new token."""
        self.tokens.append(token)


class RAGService:
    """Service for RAG (Retrieval-Augmented Generation) operations."""

    # Default prompt template for RAG
    DEFAULT_PROMPT_TEMPLATE = """You are a helpful AI assistant. Use the following pieces of context to answer the question at the end.

If you don't know the answer based on the context provided, just say "I don't have enough information to answer this question based on the provided context." Don't try to make up an answer.

Context:
{context}

Question: {question}

Answer:"""

    def __init__(
        self,
        vector_store_service=None,
        llm_service=None,
        prompt_template: Optional[str] = None,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None
    ):
        """
        Initialize RAG service.

        Args:
            vector_store_service: VectorStoreService instance
            llm_service: LLMService instance
            prompt_template: Custom prompt template
            top_k: Number of documents to retrieve
            score_threshold: Minimum similarity score
        """
        self.vector_store = vector_store_service or get_vector_store_service()
        self.llm_service = llm_service or get_llm_service()
        self.top_k = top_k or settings.rag_top_k
        self.score_threshold = score_threshold or settings.rag_score_threshold

        # Create prompt template
        template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE
        self.prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        logger.info(f"Initialized RAG service with top_k={self.top_k}")

    def _create_retriever(self) -> VectorStoreRetriever:
        """Create a retriever instance."""
        return VectorStoreRetriever(
            vector_store_service=self.vector_store,
            top_k=self.top_k,
            score_threshold=self.score_threshold
        )

    def answer_question(
        self,
        query: str,
        top_k: Optional[int] = None,
        return_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Answer a question using RAG.

        Args:
            query: User question
            top_k: Number of documents to retrieve (override default)
            return_sources: Whether to include source documents

        Returns:
            Dictionary with answer and optionally sources
        """
        try:
            logger.info(f"Processing question: {query[:100]}...")

            # Override top_k if provided
            if top_k:
                self.top_k = top_k

            # Create retriever
            retriever = self._create_retriever()

            # Get relevant documents
            docs = retriever.get_relevant_documents(query)

            if not docs:
                logger.warning("No relevant documents found")
                return {
                    "answer": "I don't have any relevant information to answer this question.",
                    "sources": [],
                    "query": query
                }

            # Build context from documents
            context = self._build_context(docs)

            # Create prompt
            prompt_text = self.prompt.format(context=context, question=query)

            # Generate answer
            llm = self.llm_service.get_llm(streaming=False)
            response = llm.invoke(prompt_text)
            answer = response.content

            # Prepare result
            result = {
                "answer": answer,
                "query": query
            }

            # Add sources if requested
            if return_sources:
                result["sources"] = self._format_sources(docs)

            logger.info(f"Successfully generated answer for query")
            return result

        except Exception as e:
            logger.error(f"Error answering question: {e}")
            raise

    async def answer_question_async(
        self,
        query: str,
        top_k: Optional[int] = None,
        return_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Async version: Answer a question using RAG.

        Args:
            query: User question
            top_k: Number of documents to retrieve (override default)
            return_sources: Whether to include source documents

        Returns:
            Dictionary with answer and optionally sources
        """
        try:
            logger.info(f"Processing async question: {query[:100]}...")

            # Override top_k if provided
            if top_k:
                self.top_k = top_k

            # Create retriever
            retriever = self._create_retriever()

            # Get relevant documents asynchronously
            docs = await retriever.aget_relevant_documents(query)

            if not docs:
                logger.warning("No relevant documents found")
                return {
                    "answer": "I don't have any relevant information to answer this question.",
                    "sources": [],
                    "query": query
                }

            # Build context from documents
            context = self._build_context(docs)

            # Create prompt
            prompt_text = self.prompt.format(context=context, question=query)

            # Generate answer using async
            llm = self.llm_service.get_llm(streaming=False)

            # Run LLM in executor for async
            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, llm.invoke, prompt_text)
            answer = response.content

            # Prepare result
            result = {
                "answer": answer,
                "query": query
            }

            # Add sources if requested
            if return_sources:
                result["sources"] = self._format_sources(docs)

            logger.info(f"Successfully generated async answer for query")
            return result

        except Exception as e:
            logger.error(f"Error answering question async: {e}")
            raise

    def answer_question_streaming(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> Iterator[Dict[str, Any]]:
        """
        Answer a question with streaming response.

        Args:
            query: User question
            top_k: Number of documents to retrieve

        Yields:
            Chunks of response with tokens and sources
        """
        try:
            logger.info(f"Processing streaming question: {query[:100]}...")

            # Override top_k if provided
            if top_k:
                self.top_k = top_k

            # Create retriever
            retriever = self._create_retriever()

            # Get relevant documents
            docs = retriever.get_relevant_documents(query)

            if not docs:
                yield {
                    "type": "answer",
                    "content": "I don't have any relevant information to answer this question.",
                    "done": True
                }
                yield {
                    "type": "sources",
                    "sources": []
                }
                return

            # Build context
            context = self._build_context(docs)

            # Create prompt
            prompt_text = self.prompt.format(context=context, question=query)

            # Stream answer
            for token in self.llm_service.generate_streaming(prompt_text):
                yield {
                    "type": "token",
                    "content": token,
                    "done": False
                }

            # Send completion signal
            yield {
                "type": "answer",
                "content": "",
                "done": True
            }

            # Send sources
            yield {
                "type": "sources",
                "sources": self._format_sources(docs)
            }

            logger.info("Completed streaming answer")

        except Exception as e:
            logger.error(f"Error in streaming answer: {e}")
            raise

    def _build_context(self, documents: List[Document]) -> str:
        """
        Build context string from documents.

        Args:
            documents: List of Document objects

        Returns:
            Formatted context string
        """
        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"[{i}] {doc.page_content}")

        return "\n\n".join(context_parts)

    def _format_sources(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Format source documents for response.

        Args:
            documents: List of Document objects

        Returns:
            List of formatted source dictionaries
        """
        sources = []
        for doc in documents:
            source = {
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "score": doc.metadata.get("score"),
                "metadata": {k: v for k, v in doc.metadata.items() if k not in ["score"]}
            }
            sources.append(source)

        return sources


# Global RAG service instance
_rag_service: Optional[RAGService] = None


def get_rag_service(
    vector_store_service=None,
    llm_service=None,
    top_k: Optional[int] = None
) -> RAGService:
    """
    Get or create the global RAG service instance.

    Args:
        vector_store_service: Optional VectorStoreService override
        llm_service: Optional LLMService override
        top_k: Optional top_k override

    Returns:
        RAGService instance
    """
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService(
            vector_store_service=vector_store_service,
            llm_service=llm_service,
            top_k=top_k
        )
    return _rag_service
