from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Qdrant Configuration
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "documents"

    # Embedding Model
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # API Configuration
    api_title: str = "RAG Application API"
    api_version: str = "1.0.0"
    api_description: str = "FastAPI application for RAG with Qdrant"

    # CORS Configuration
    cors_origins: str = "http://localhost:3000,http://localhost:8000"

    # Application Settings
    debug: bool = True
    log_level: str = "INFO"

    @property
    def cors_origins_list(self) -> List[str]:
        """Parse CORS origins from comma-separated string."""
        return [origin.strip() for origin in self.cors_origins.split(",")]

    class Config:
        env_file = ".env"
        case_sensitive = False


# Create global settings instance
settings = Settings()
