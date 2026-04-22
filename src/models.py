import os
from typing import Any, TypeVar

from langchain.chat_models.base import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from esg_agent.config import ESGCategory, ModelType, settings

T = TypeVar("T", bound=BaseModel)


class ESGEvent(BaseModel):
    """A single ESG-relevant event extracted from a sustainability report."""

    company: str = Field(description="The name of the company involved in the event.")
    event: str = Field(description="A concise description of the event or action.")
    category: ESGCategory = Field(description="The ESG category this event belongs to.")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score for the extraction (0.0 to 1.0).",
    )
    source_excerpt: str = Field(
        description="The verbatim excerpt from the source text supporting this event."
    )
    validated: bool = Field(
        default=False,
        description="Whether the event passed the validation layer.",
    )


class ESGExtractionResult(BaseModel):
    """Result of extracting ESG events from a report."""

    events: list[ESGEvent] = Field(default_factory=list)
    raw_context: str = Field(
        default="",
        description="The retrieved RAG context used for extraction.",
    )

    @classmethod
    def empty(cls) -> "ESGExtractionResult":
        return cls(events=[], raw_context="")


class ValidationResult(BaseModel):
    """Result of validating a single ESGEvent against its source."""

    event: ESGEvent
    is_grounded: bool = Field(
        description="True if the event is supported by the source excerpt."
    )
    grounding_explanation: str = Field(
        description="Explanation of why the event is or is not grounded."
    )


class ModelFactory:
    """Factory for creating LLM chat models."""

    @staticmethod
    def create_model(
        model_type: ModelType,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> BaseChatModel:
        if model_type in [
            ModelType.OPENAI_GPT4O,
            ModelType.OPENAI_GPT4O_MINI,
            ModelType.OPENAI_GPT5_MINI,
        ]:
            return ChatOpenAI(
                model=model_type.value,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

        if model_type == ModelType.GEMINI_2_5_FLASH:
            from langchain_google_genai import ChatGoogleGenerativeAI

            google_api_key = kwargs.pop("google_api_key", settings.google_api_key)
            if google_api_key is None:
                raise ValueError("google_api_key is required for Gemini models")
            return ChatGoogleGenerativeAI(
                model=model_type.value,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=google_api_key,
                **kwargs,
            )

        raise ValueError(f"Unsupported model type: {model_type}")

    @staticmethod
    def create_structured_model(
        model_type: ModelType,
        schema: type[T],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> Any:
        """Create a model configured for structured output matching the given schema."""
        base = ModelFactory.create_model(
            model_type, temperature=temperature, max_tokens=max_tokens, **kwargs
        )
        try:
            return base.with_structured_output(schema, method="json_mode")
        except TypeError:
            return base.with_structured_output(schema)

    @staticmethod
    def get_model_type() -> ModelType:
        model_str = os.getenv("ESG_DEFAULT_MODEL", settings.default_model)
        try:
            return ModelType(model_str)
        except ValueError:
            return ModelType.OPENAI_GPT4O_MINI
