"""Schema definitions for structured output responses."""
from pydantic import BaseModel, Field
from typing import Optional, Literal


class MatchReason(BaseModel):
    """Explanation of why a signal contributed to the match."""
    signal: str  # vibe, cuisine, price, features
    importance: Literal["primary", "secondary", "minor"]
    query_wanted: str
    restaurant_has: str
    score: float = Field(ge=0, le=1)


class EnrichedRestaurantMatch(BaseModel):
    """Enriched restaurant match with all fields needed for structured display."""
    # Basic fields
    id: str
    name: str
    cuisine: str
    price_level: str
    price_level_curated: Optional[str] = None
    region: str
    city: Optional[str] = None
    formatted_address: Optional[str] = None
    rating: float
    features: dict[str, bool] = Field(default_factory=dict)
    vibe_summary: str
    vibe_formality: Optional[str] = None  # very_casual, casual, smart_casual, upscale, fine_dining
    vibe_noise_level: Optional[str] = None  # quiet, moderate, lively, loud
    vibe_atmosphere_tags: list[str] = Field(default_factory=list)  # romantic, trendy, etc.
    vibe_best_for: list[str] = Field(default_factory=list)  # groups, dinner, etc.
    
    # Additional fields for structured display
    highlights: str = ""
    details: str = ""
    editorial_summary: str = ""
    top_menu_items: list[str] = Field(default_factory=list)
    photos_urls: list[str] = Field(default_factory=list)
    restaurant_photos_urls: list[str] = Field(default_factory=list)
    video_urls: list[str] = Field(default_factory=list)
    
    # Scores
    final_score: float = Field(ge=0, le=1)
    vibe_score: float = Field(ge=0, le=1)
    cuisine_score: float = Field(ge=0, le=1)
    price_score: float = Field(ge=0, le=1)
    feature_score: float = Field(ge=0, le=1)


class StructuredSearchResponse(BaseModel):
    """Structured search response for conversational interface."""
    success: bool
    top_match: Optional[EnrichedRestaurantMatch] = None
    alternatives: list[EnrichedRestaurantMatch] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "medium"
    caveats: list[str] = Field(default_factory=list)
    match_reasons: list[MatchReason] = Field(default_factory=list)
    query: str = ""  # Store the original query for context in formatting
    explanation: str = ""  # LLM-generated natural language explanation

