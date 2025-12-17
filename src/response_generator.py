from pathlib import Path
from openai import AsyncOpenAI
from .schemas import ParsedQuery, AnaResponse, RestaurantMatch, MatchReason
from .fusion import ScoredRestaurant


class ResponseGenerator:
    """Generate natural language responses using LLM."""
    
    def __init__(self, client: AsyncOpenAI | None = None):
        self.client = client or AsyncOpenAI()
        self.system_prompt = self._load_prompt()
    
    def _load_prompt(self) -> str:
        prompt_path = Path(__file__).parent.parent / "prompts" / "response_generator.txt"
        if prompt_path.exists():
            return prompt_path.read_text()
        return "Generate a helpful restaurant recommendation response."
    
    def _scored_to_match(self, scored: ScoredRestaurant) -> RestaurantMatch:
        """Convert ScoredRestaurant to RestaurantMatch."""
        return RestaurantMatch(
            id=scored.restaurant.id,
            name=scored.restaurant.name,
            cuisine=scored.restaurant.cuisine,
            price_level=scored.restaurant.price_level,
            region=scored.restaurant.region,
            rating=scored.restaurant.rating,
            vibe_summary=scored.restaurant.vibe.vibe_summary,
            final_score=scored.final_score,
            vibe_score=scored.vibe_score,
            cuisine_score=scored.cuisine_score,
            price_score=scored.price_score,
            feature_score=scored.feature_score,
        )
    
    def _generate_match_reasons(self, scored: ScoredRestaurant, parsed_query: ParsedQuery) -> list[MatchReason]:
        """Generate match reasons based on scores."""
        reasons = []
        
        # Sort scores to determine importance
        scores = [
            ("vibe", scored.vibe_score, parsed_query.weights.vibe),
            ("cuisine", scored.cuisine_score, parsed_query.weights.cuisine),
            ("price", scored.price_score, parsed_query.weights.price),
            ("features", scored.feature_score, parsed_query.weights.features),
        ]
        scores.sort(key=lambda x: x[1] * x[2], reverse=True)
        
        for i, (signal, score, weight) in enumerate(scores):
            if score < 0.3:
                continue
                
            importance = "primary" if i == 0 else ("secondary" if i == 1 else "minor")
            
            if signal == "vibe":
                reasons.append(MatchReason(
                    signal=signal,
                    importance=importance,
                    query_wanted=parsed_query.semantic_query[:100],
                    restaurant_has=scored.restaurant.vibe.vibe_summary[:100],
                    score=score,
                ))
            elif signal == "cuisine":
                reasons.append(MatchReason(
                    signal=signal,
                    importance=importance,
                    query_wanted=", ".join(parsed_query.preferences.cuisine) or "any",
                    restaurant_has=scored.restaurant.cuisine,
                    score=score,
                ))
            elif signal == "price":
                reasons.append(MatchReason(
                    signal=signal,
                    importance=importance,
                    query_wanted=", ".join(parsed_query.preferences.price) or "any",
                    restaurant_has=scored.restaurant.price_level,
                    score=score,
                ))
            elif signal == "features":
                reasons.append(MatchReason(
                    signal=signal,
                    importance=importance,
                    query_wanted=", ".join(parsed_query.preferences.features + parsed_query.preferences.atmosphere) or "any",
                    restaurant_has=", ".join(scored.restaurant.vibe.atmosphere_tags[:3]),
                    score=score,
                ))
        
        return reasons
    
    def _determine_confidence(self, top_score: float, num_results: int) -> str:
        """Determine confidence level based on match quality."""
        if top_score >= 0.8 and num_results >= 1:
            return "high"
        elif top_score >= 0.6:
            return "medium"
        else:
            return "low"
    
    async def generate(
        self,
        parsed_query: ParsedQuery,
        ranked_results: list[ScoredRestaurant],
    ) -> AnaResponse:
        """Generate complete response with explanation."""
        if not ranked_results:
            return AnaResponse(
                success=False,
                explanation="I couldn't find any restaurants matching your criteria. Try broadening your search?",
                confidence="low",
                caveats=["No matching restaurants found"],
            )
        
        top_match = ranked_results[0]
        alternatives = ranked_results[1:4]  # Up to 3 alternatives
        
        # Generate natural language explanation
        user_prompt = f"""
Query: {parsed_query.raw_query}

Top Match: {top_match.restaurant.name}
- Cuisine: {top_match.restaurant.cuisine}
- Price: {top_match.restaurant.price_level}
- Vibe: {top_match.restaurant.vibe.vibe_summary}
- Score: {top_match.final_score:.2f}

Generate a warm, helpful explanation for why this restaurant is perfect for them.
"""
        
        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=200,
        )
        
        explanation = response.choices[0].message.content.strip()
        
        return AnaResponse(
            success=True,
            top_match=self._scored_to_match(top_match),
            alternatives=[self._scored_to_match(alt) for alt in alternatives],
            match_reasons=self._generate_match_reasons(top_match, parsed_query),
            explanation=explanation,
            confidence=self._determine_confidence(top_match.final_score, len(ranked_results)),
        )

