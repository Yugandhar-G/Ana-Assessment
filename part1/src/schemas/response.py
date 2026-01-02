from pydantic import BaseModel, Field
from typing import Optional, Literal


class MatchReason(BaseModel):
    """Explanation of why a signal contributed to the match."""
    signal: str  # vibe, cuisine, price, features
    importance: Literal["primary", "secondary", "minor"]
    query_wanted: str
    restaurant_has: str
    score: float = Field(ge=0, le=1)


class RestaurantMatch(BaseModel):
    """A matched restaurant with scoring details."""
    id: str
    name: str
    cuisine: str
    price_level: str
    price_level_curated: Optional[str] = None
    region: str
    city: Optional[str] = None
    formatted_address: Optional[str] = None
    location_raw: Optional[str] = None
    state: Optional[str] = None
    zipcode: Optional[str] = None
    country: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    rating: float
    features: dict[str, bool] = Field(default_factory=dict)
    vibe_summary: str
    
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
    video_urls: list[str] = Field(default_factory=list, description="Video URLs associated with this restaurant from video metadata")
    reviews: Optional[str] = None
    
    # Metadata
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
    live_music_curated: Optional[bool] = None
    live_music_google: Optional[bool] = None
    
    # Scores
    final_score: float = Field(ge=0, le=1)
    
    # Score breakdown
    vibe_score: float = Field(ge=0, le=1)
    cuisine_score: float = Field(ge=0, le=1)
    price_score: float = Field(ge=0, le=1)
    feature_score: float = Field(ge=0, le=1)


class AnaResponse(BaseModel):
    """Complete response from Ana AI."""
    success: bool
    top_match: Optional[RestaurantMatch] = None
    alternatives: list[RestaurantMatch] = Field(default_factory=list)
    match_reasons: list[MatchReason] = Field(default_factory=list)
    explanation: str = ""
    confidence: Literal["high", "medium", "low"] = "medium"
    caveats: list[str] = Field(default_factory=list)

