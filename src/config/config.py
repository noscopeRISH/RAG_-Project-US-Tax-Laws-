"""Configuration module for Agentic RAG System."""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


from langchain_mistralai import ChatMistralAI


class Config:
    """Configuration class for the RAG system."""

    # API Key
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

    # Model Configuration
    LLM_MODEL = "mistral-large-latest"

    # Document Processing (approx. 500-800 tokens with 10% overlap)
    CHUNK_SIZE = 2500
    CHUNK_OVERLAP = 250

    # default urls
    DEFAULT_URLS = [
        "https://www.irs.gov/newsroom/irs-guidance",
        "https://www.irs.gov/taxpayer-bill-of-rights",
    ]

    @classmethod
    def get_llm(cls):
        """Initialize and return the LLM."""
        if not cls.MISTRAL_API_KEY:
            raise ValueError(
                "MISTRAL_API_KEY not found. Add it to your .env file."
            )
        return ChatMistralAI(
            model=cls.LLM_MODEL,
            api_key=cls.MISTRAL_API_KEY,
        )