from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Answer(BaseModel):
    """The structured final answer. The agent must submit this via the submit_answer tool.

    Delivering the answer through a tool call (instead of provider-specific JSON modes)
    works on every OpenAI-compatible provider, including free ones.
    """

    answer: str = Field(description="The final answer in plain language, 1-4 sentences.")
    value: str | None = Field(
        default=None,
        description=(
            "The key result as a plain string with no units or extra words "
            "(a number like '37.2', an id like 'M-05', or a date like '2026-03-04'). "
            "Omit it if the question cannot be answered from the data."
        ),
    )
    confidence: Literal["high", "medium", "low", "none"] = Field(
        description="'none' means the data cannot answer the question."
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="The facts or SQL queries this answer is based on.",
    )
    caveats: list[str] = Field(
        default_factory=list,
        description="Data-quality issues or assumptions the reader should know about.",
    )
