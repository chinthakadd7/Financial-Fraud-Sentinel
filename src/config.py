"""
Configuration Management for Fraud Detection System.

Implements 12-factor app methodology with environment-based configuration.
Uses Pydantic Settings for validation and type safety.
"""

import os
import json
from pathlib import Path
from typing import Optional, Annotated
from pydantic_settings import BaseSettings, SettingsConfigDict, NoDecode
from pydantic import Field, field_validator


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Supports .env file loading with sensible defaults for development.
    Override in production using environment variables.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Application Settings
    APP_NAME: str = Field(
        default="Financial Fraud Sentinel",
        description="Application name"
    )
    VERSION: str = Field(
        default="1.0.0",
        description="API version"
    )
    ENVIRONMENT: str = Field(
        default="development",
        description="Environment: development, staging, production"
    )
    DEBUG: bool = Field(
        default=False,
        description="Enable debug mode"
    )
    
    # API Configuration
    API_HOST: str = Field(
        default="0.0.0.0",
        description="API server host"
    )
    API_PORT: int = Field(
        default=8000,
        description="API server port"
    )
    API_PREFIX: str = Field(
        default="",
        description="API route prefix (e.g., /api/v1)"
    )
    ALLOWED_ORIGINS: Annotated[list[str], NoDecode] = Field(
        default=["http://localhost:10101", "http://127.0.0.1:10101"],
        description="CORS allowed origins"
    )
    
    # Dashboard Configuration
    DASHBOARD_URL: str = Field(
        default="http://localhost:10101",
        description="Dashboard URL for CORS"
    )
    
    # Model Configuration
    MODEL_PATH: str = Field(
        default="models_artifacts/XGBoost_1_AutoML_1_20260209_165338.zip",
        description="Path to H2O MOJO model file"
    )
    MODEL_NAME: str = Field(
        default="XGBoost_1_AutoML_1_20260209_165338",
        description="Model identifier"
    )
    
    # H2O Configuration
    H2O_MEMORY_GB: int = Field(
        default=2,
        description="H2O server memory allocation in GB",
        ge=1,
        le=32
    )
    H2O_PORT: int = Field(
        default=54321,
        description="H2O server port"
    )
    H2O_NTHREADS: int = Field(
        default=-1,
        description="H2O number of threads (-1 = all available)"
    )
    H2O_MAX_MEM_SIZE: Optional[str] = Field(
        default=None,
        description="H2O max memory size (e.g., '2g'). Overrides H2O_MEMORY_GB if set."
    )
    
    # Logging Configuration
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL"
    )
    LOG_FORMAT: str = Field(
        default="json",
        description="Log format: json or text"
    )
    LOG_FILE: Optional[str] = Field(
        default=None,
        description="Path to log file (None = stdout only)"
    )
    
    # Performance Configuration
    ENABLE_CACHE: bool = Field(
        default=False,
        description="Enable prediction caching (not implemented)"
    )
    CACHE_TTL_SECONDS: int = Field(
        default=300,
        description="Cache time-to-live in seconds"
    )
    REQUEST_TIMEOUT_SECONDS: int = Field(
        default=30,
        description="API request timeout"
    )
    
    # Security Configuration
    ENABLE_RATE_LIMIT: bool = Field(
        default=False,
        description="Enable rate limiting (not implemented)"
    )
    RATE_LIMIT_REQUESTS: int = Field(
        default=100,
        description="Max requests per minute per IP"
    )
    
    @field_validator("ENVIRONMENT")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment is one of the allowed values."""
        allowed = ["development", "staging", "production"]
        if v.lower() not in allowed:
            raise ValueError(f"Environment must be one of {allowed}")
        return v.lower()
    
    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is valid."""
        allowed = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in allowed:
            raise ValueError(f"Log level must be one of {allowed}")
        return v_upper
    
    @field_validator("MODEL_PATH")
    @classmethod
    def validate_model_path(cls, v: str) -> str:
        """Ensure model path exists or provide helpful error."""
        # Convert to Path for cross-platform compatibility
        model_path = Path(v)
        
        # If relative path, resolve from project root
        if not model_path.is_absolute():
            # Try to find project root (where models_artifacts/ should be)
            current = Path.cwd()
            # Check if we're in src/ or deeper
            if "src" in current.parts:
                # Navigate up to project root
                while current.name != "financial-fraud-sentinel" and current.parent != current:
                    current = current.parent
                model_path = current / v
            else:
                model_path = current / v
        
        # In production, model must exist. In dev, just warn.
        if not model_path.exists():
            # Don't fail - model might be mounted in Docker
            pass
        
        return str(model_path)

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def validate_allowed_origins(cls, v):
        """Support JSON array or comma-separated origins from env."""
        if isinstance(v, list):
            return [str(origin).strip() for origin in v if str(origin).strip()]

        if isinstance(v, str):
            value = v.strip()
            if not value:
                return []

            if value.startswith("["):
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, list):
                        return [str(origin).strip() for origin in parsed if str(origin).strip()]
                except json.JSONDecodeError as exc:
                    raise ValueError("ALLOWED_ORIGINS must be a valid JSON array or comma-separated string") from exc

            return [origin.strip() for origin in value.split(",") if origin.strip()]

        raise ValueError("ALLOWED_ORIGINS must be a list or string")
    
    @property
    def h2o_memory_str(self) -> str:
        """Get H2O memory setting as string (e.g., '2g')."""
        if self.H2O_MAX_MEM_SIZE:
            return self.H2O_MAX_MEM_SIZE
        return f"{self.H2O_MEMORY_GB}g"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.ENVIRONMENT == "development"


# Global settings instance
# Can be imported and used throughout the application
settings = Settings()


def get_settings() -> Settings:
    """
    Dependency injection function for FastAPI.
    
    Returns the global settings instance.
    Can be used in FastAPI routes with Depends(get_settings).
    """
    return settings
