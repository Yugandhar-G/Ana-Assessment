from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class RestaurantVibe(BaseModel):
    """Enriched vibe profile generated from restaurant data."""
    formality: str  # very_casual, casual, smart_casual, upscale, fine_dining
    noise_level: str  # quiet, moderate, lively, loud
    atmosphere_tags: list[str] = Field(default_factory=list)  # romantic, trendy, hole-in-the-wall
    best_for: list[str] = Field(default_factory=list)  # date night, groups, solo, business
    vibe_summary: str  # Natural language description for embeddings


class Restaurant(BaseModel):
    """Restaurant with enriched vibe profile."""
    # Core identity
    id: str
    name: str
    google_place_id: Optional[str] = None
    
    # Structured fields
    cuisine: str
    price_level: str  # $, $$, $$$, $$$$
    region: str
    rating: float = Field(ge=0, le=5)
    
    # Boolean features
    features: dict[str, bool] = Field(default_factory=dict)
    
    # Text fields
    highlights: str = ""
    details: str = ""
    editorial_summary: str = ""
    
    # Enriched vibe profile
    vibe: RestaurantVibe
    
    # Metadata
    last_updated: datetime = Field(default_factory=datetime.now)

