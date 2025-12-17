from pydantic import BaseModel, Field
from typing import Optional


class MustNotFilters(BaseModel):
    """Hard exclusions - restaurants matching these are filtered out."""
    formality: list[str] = Field(default_factory=list)  # ["upscale", "fine_dining"]
    price: list[str] = Field(default_factory=list)  # ["$$$$"]
    cuisine: list[str] = Field(default_factory=list)
    features: list[str] = Field(default_factory=list)  # ["loud", "crowded"]


class Preferences(BaseModel):
    """Soft preferences - boost matching restaurants but don't exclude."""
    cuisine: list[str] = Field(default_factory=list)
    price: list[str] = Field(default_factory=list)
    features: list[str] = Field(default_factory=list)
    atmosphere: list[str] = Field(default_factory=list)


class SignalWeights(BaseModel):
    """Dynamic weights for score fusion, based on query emphasis."""
    vibe: float = Field(default=0.5, ge=0, le=1)
    cuisine: float = Field(default=0.25, ge=0, le=1)
    price: float = Field(default=0.15, ge=0, le=1)
    features: float = Field(default=0.1, ge=0, le=1)


class ParsedQuery(BaseModel):
    """Structured representation of user's search intent."""
    raw_query: str
    semantic_query: str  # Expanded query for vector search
    must_not: MustNotFilters = Field(default_factory=MustNotFilters)
    preferences: Preferences = Field(default_factory=Preferences)
    weights: SignalWeights = Field(default_factory=SignalWeights)
    location: Optional[str] = None

