"""
Application configuration settings.
"""

import os
from pathlib import Path
from typing import List

from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # API Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    
    # CORS Settings
    ALLOWED_HOSTS: List[str] = ["*"]
    
    # Model Configuration
    MODEL_PATH: str = "svc_model.pkl"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Application
    APP_NAME: str = "Iris Classification API"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "A FastAPI application for classifying iris flowers using machine learning"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
