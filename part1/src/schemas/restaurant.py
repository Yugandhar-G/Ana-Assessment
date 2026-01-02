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
    city: Optional[str] = None
    formatted_address: Optional[str] = None
    location_raw: Optional[str] = None
    state: Optional[str] = None
    zipcode: Optional[str] = None
    country: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    rating: float = Field(ge=0, le=5)
    price_level_curated: Optional[str] = None
    
    # Business details
    business_status: Optional[str] = None
    national_phone: Optional[str] = None
    international_phone: Optional[str] = None
    website_uri: Optional[str] = None
    google_maps_uri: Optional[str] = None
    opening_hours_text: Optional[str] = None
    is_open_now: Optional[bool] = None
    serves_meal_times: list[str] = Field(default_factory=list)
    
    # Media
    photos_urls: list[str] = Field(default_factory=list)
    restaurant_photos_urls: list[str] = Field(default_factory=list)
    reviews: Optional[str] = None
    
    # Boolean features
    features: dict[str, bool] = Field(default_factory=dict)
    live_music_curated: Optional[bool] = None
    live_music_google: Optional[bool] = None
    
    # Text fields
    highlights: str = ""
    details: str = ""
    editorial_summary: str = ""
    top_menu_items: list[str] = Field(default_factory=list)  # Popular menu items if available
    
    # Enriched vibe profile
    vibe: RestaurantVibe
    
    # Metadata
    last_updated: datetime = Field(default_factory=datetime.now)
    payment_options: Optional[str] = None
    parking_options: Optional[str] = None
    restroom: Optional[str] = None
    match_status: Optional[str] = None
    match_confidence: Optional[float] = None
    name_similarity: Optional[float] = None
    data_completeness_score: Optional[float] = None
    google_matched_name: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

