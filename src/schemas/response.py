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
    region: str
    rating: float
    vibe_summary: str
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

