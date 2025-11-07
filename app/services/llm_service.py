from typing import Optional, Iterator
import logging
from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from app.config.settings import settings

logger = logging.getLogger(__name__)


class StreamingCallbackHandler(BaseCallbackHandler):
    """Custom callback handler for streaming tokens."""

    def __init__(self):
        self.tokens = []

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Handle new token from LLM."""
        self.tokens.append(token)

    def get_tokens(self) -> list:
        """Get all collected tokens."""
        return self.tokens


class LLMService:
    """Service for interacting with LLM through LangChain."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ):
        """
        Initialize LLM service.

        Args:
            api_key: OpenAI API key (default: from settings)
            model: Model name (default: from settings)
            temperature: Temperature for generation (default: from settings)
            max_tokens: Maximum tokens to generate (default: from settings)
        """
        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.llm_model
        self.temperature = temperature if temperature is not None else settings.llm_temperature
        self.max_tokens = max_tokens or settings.llm_max_tokens

        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Please set OPENAI_API_KEY in environment variables."
            )

        logger.info(f"Initializing LLM service with model: {self.model}")

    def get_llm(self, streaming: bool = False) -> ChatOpenAI:
        """
        Get LangChain ChatOpenAI instance.

        Args:
            streaming: Whether to enable streaming mode

        Returns:
            ChatOpenAI instance
        """
        try:
            llm = ChatOpenAI(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                openai_api_key=self.api_key,
                streaming=streaming
            )
            return llm
        except Exception as e:
            logger.error(f"Error creating LLM instance: {e}")
            raise

    def generate(self, prompt: str) -> str:
        """
        Generate response from LLM.

        Args:
            prompt: Input prompt

        Returns:
            Generated text
        """
        try:
            llm = self.get_llm(streaming=False)
            response = llm.invoke(prompt)
            return response.content
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    def generate_streaming(self, prompt: str) -> Iterator[str]:
        """
        Generate streaming response from LLM.

        Args:
            prompt: Input prompt

        Yields:
            Tokens as they are generated
        """
        try:
            llm = self.get_llm(streaming=True)

            # Use streaming
            for chunk in llm.stream(prompt):
                if hasattr(chunk, 'content'):
                    yield chunk.content
        except Exception as e:
            logger.error(f"Error in streaming generation: {e}")
            raise


# Global LLM service instance
_llm_service: Optional[LLMService] = None


def get_llm_service(
    api_key: Optional[str] = None,
    model: Optional[str] = None
) -> LLMService:
    """
    Get or create the global LLM service instance.

    Args:
        api_key: Optional API key override
        model: Optional model override

    Returns:
        LLMService instance
    """
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService(api_key=api_key, model=model)
    return _llm_service
