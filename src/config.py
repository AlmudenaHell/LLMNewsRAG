import os
from enum import Enum

from pydantic import Field
from pydantic_settings import BaseSettings


class ModelType(str, Enum):
    OPENAI_GPT4O = "gpt-4o"
    OPENAI_GPT4O_MINI = "gpt-4o-mini"
    OPENAI_GPT5_MINI = "gpt-5-mini-2025-08-07"
    GEMINI_2_5_FLASH = "gemini-2.5-flash"


class TaskType(str, Enum):
    extraction = "extraction"
    validation = "validation"


class ESGCategory(str, Enum):
    environmental = "Environmental"
    social = "Social"
    governance = "Governance"
    expansion = "Expansion"
    controversy = "Controversy"
    commitment = "Commitment"
    other = "Other"


class Settings(BaseSettings):
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    google_api_key: str | None = Field(default=None, alias="GOOGLE_API_KEY")
    default_model: str = Field(default="gpt-4o-mini", alias="ESG_DEFAULT_MODEL")
    max_web_searches: int = Field(default=3, alias="ESG_MAX_WEB_SEARCHES")
    rag_top_k: int = Field(default=5, alias="ESG_RAG_TOP_K")
    use_rate_limiter: bool = Field(default=False, alias="USE_RATE_LIMITER")

    model_config = {"populate_by_name": True}


settings = Settings()
