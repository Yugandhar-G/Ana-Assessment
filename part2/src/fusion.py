"""
Score fusion algorithms for combining multiple signal scores.
"""
import math
import re
import logging
from dataclasses import dataclass
from .schemas import Restaurant, ParsedQuery, SignalWeights

logger = logging.getLogger(__name__)


def has_award(restaurant: Restaurant) -> bool:
    """Check if restaurant has awards mentioned in highlights or details."""
    award_keywords = [
        r"award",
        r"winner",
        r"best\s+\w+\s+(cuisine|restaurant|service|menu)",
        r"gold\s+award",
        r"silver\s+award",
        r"honorable\s+mention",
        r"recognition",
        r"prize",
        r"accolade",
        r"'aipono",  # Common award name in the data
    ]
    
    text_to_check = f"{restaurant.highlights} {restaurant.details}".lower()
    
    for pattern in award_keywords:
        if re.search(pattern, text_to_check, re.IGNORECASE):
            return True
    
    return False


def is_traditional_restaurant(restaurant: Restaurant) -> bool:
    """Check if restaurant is a traditional restaurant (not ghost kitchen, catering, or food truck)."""
    text_to_check = f"{restaurant.name} {restaurant.highlights} {restaurant.details} {restaurant.editorial_summary}".lower()
    
    # Non-traditional restaurant indicators
    non_traditional_indicators = [
        r"ghost\s+kitchen",
        r"catering\s+(service|only|model)",
        r"food\s+truck",
        r"pickup.*delivery.*only",
        r"no\s+dine.*in",
    ]
    
    for pattern in non_traditional_indicators:
        if re.search(pattern, text_to_check, re.IGNORECASE):
            return False
    
    return True


def has_best_mention(restaurant: Restaurant, cuisine: str = None) -> float:
    """Check if restaurant is mentioned as 'best' for cuisine or in general.
    Returns a boost score (0.0 to 1.0) based on how strong the 'best' mention is.
    """
    text_to_check = f"{restaurant.highlights} {restaurant.details} {restaurant.editorial_summary}".lower()
    
    # Strong "best" mentions
    if cuisine:
        cuisine_lower = cuisine.lower()
        # "best [cuisine] restaurant" or "best [cuisine]"
        strong_patterns = [
            rf"best\s+{re.escape(cuisine_lower)}\s+restaurant",
            rf"considered\s+the\s+best\s+{re.escape(cuisine_lower)}",
            rf"best\s+{re.escape(cuisine_lower)}\s+on\s+\w+",  # "best indian on maui"
        ]
        for pattern in strong_patterns:
            if re.search(pattern, text_to_check, re.IGNORECASE):
                return 1.0  # Maximum boost
    
    # General "best restaurant" mentions
    if re.search(r"best\s+(restaurant|place|spot)", text_to_check, re.IGNORECASE):
        return 0.5  # Moderate boost
    
    # "Best" in context of cuisine (weaker)
    if re.search(r"best\s+\w+", text_to_check, re.IGNORECASE):
        return 0.2  # Small boost
    
    return 0.0


def get_award_level(restaurant: Restaurant) -> float:
    """
    Get award level boost (0.0 to 1.0).
    Gold Award = 1.0, Silver Award = 0.8, Honorable Mention = 0.5, Other = 0.3
    """
    text_to_check = f"{restaurant.highlights} {restaurant.details}".lower()
    
    if re.search(r"gold\s+award", text_to_check, re.IGNORECASE):
        return 1.0
    elif re.search(r"silver\s+award", text_to_check, re.IGNORECASE):
        return 0.8
    elif re.search(r"honorable\s+mention", text_to_check, re.IGNORECASE):
        return 0.5
    elif has_award(restaurant):
        return 0.3
    
    return 0.0


@dataclass
class ScoredRestaurant:
    """Restaurant with all signal scores and final fused score."""
    restaurant: Restaurant
    vibe_score: float
    cuisine_score: float
    price_score: float
    feature_score: float
    final_score: float


class ScoreFusion:
    """Simple weighted sum fusion (legacy, kept for backward compatibility)."""
    
    def fuse(
        self,
        restaurant: Restaurant,
        vibe_score: float,
        cuisine_score: float,
        price_score: float,
        feature_score: float,
        weights: SignalWeights,
        parsed_query: ParsedQuery | None = None,
    ) -> ScoredRestaurant:
        """Compute weighted fusion of all scores."""
        final_score = (
            weights.vibe * vibe_score +
            weights.cuisine * cuisine_score +
            weights.price * price_score +
            weights.features * feature_score
        )
        
        # Ensure all scores are in valid range [0, 1] to satisfy Pydantic schema
        clamped_vibe_score = max(0.0, min(1.0, vibe_score))
        clamped_cuisine_score = max(0.0, min(1.0, cuisine_score))
        clamped_price_score = max(0.0, min(1.0, price_score))
        clamped_feature_score = max(0.0, min(1.0, feature_score))
        clamped_final_score = max(0.0, min(1.0, final_score))
        
        return ScoredRestaurant(
            restaurant=restaurant,
            vibe_score=clamped_vibe_score,
            cuisine_score=clamped_cuisine_score,
            price_score=clamped_price_score,
            feature_score=clamped_feature_score,
            final_score=clamped_final_score,
        )
    
    def rank(self, scored_restaurants: list[ScoredRestaurant]) -> list[ScoredRestaurant]:
        """Rank restaurants by final score, descending."""
        return sorted(scored_restaurants, key=lambda x: x.final_score, reverse=True)
    
    def rank_with_award_priority(
        self, 
        scored_restaurants: list[ScoredRestaurant], 
        top_n: int = 10,
        parsed_query: ParsedQuery | None = None
    ) -> list[ScoredRestaurant]:
        """
        Rank restaurants with award winners AND primary cuisine matches prioritized among top N.
        Primary cuisine matches get prioritized over secondary cuisine matches, even if they have awards.
        """
        if len(scored_restaurants) <= top_n:
            # Even if we have <= top_n results, still apply primary cuisine prioritization
            # Sort all by score first
            sorted_all = sorted(scored_restaurants, key=lambda x: x.final_score, reverse=True)
            top_n_results = sorted_all[:top_n]
            rest = []
        else:
            # Sort all by score first
            sorted_all = sorted(scored_restaurants, key=lambda x: x.final_score, reverse=True)
            
            # Separate top N
            top_n_results = sorted_all[:top_n]
            rest = sorted_all[top_n:]
        
        # Sort all by score first
        sorted_all = sorted(scored_restaurants, key=lambda x: x.final_score, reverse=True)
        
        # Separate top N
        top_n_results = sorted_all[:top_n]
        rest = sorted_all[top_n:]
        
        # Re-rank top N with priority: primary cuisine matches > (award winners + secondary cuisine) > others
        # Primary cuisine matches (cuisine_score >= 0.95) are prioritized even over award winners
        primary_cuisine = []
        award_winners_secondary = []
        others = []
        
        for scored in top_n_results:
            is_primary_cuisine = False
            if parsed_query and parsed_query.preferences.cuisine:
                # Check if cuisine_score indicates primary match (>= 0.95) vs secondary (< 0.95)
                is_primary_cuisine = scored.cuisine_score >= 0.95
            
            is_award = has_award(scored.restaurant)
            
            if is_primary_cuisine:
                primary_cuisine.append(scored)
                logger.debug(f"RANKING: {scored.restaurant.name} → PRIMARY_CUISINE (cuisine_score={scored.cuisine_score:.2f})")
            elif is_award or (parsed_query and parsed_query.preferences.cuisine and 0.7 <= scored.cuisine_score < 0.95):
                # Award winners or secondary cuisine matches
                award_winners_secondary.append(scored)
                logger.debug(f"RANKING: {scored.restaurant.name} → AWARD_SECONDARY (cuisine_score={scored.cuisine_score:.2f}, has_award={is_award})")
            else:
                others.append(scored)
                logger.debug(f"RANKING: {scored.restaurant.name} → OTHERS (cuisine_score={scored.cuisine_score:.2f})")
        
        # Sort primary cuisine matches by: 
        # 1. Traditional restaurant type (True > False) - prioritize actual restaurants over ghost kitchens/catering
        # 2. "Best" mention boost (higher is better)
        # 3. final_score (desc)
        # 4. rating (desc)
        # 5. award level (desc)
        requested_cuisine = parsed_query.preferences.cuisine[0] if (parsed_query and parsed_query.preferences.cuisine) else None
        primary_cuisine.sort(
            key=lambda x: (
                is_traditional_restaurant(x.restaurant),  # True (1) > False (0) - traditional restaurants first
                has_best_mention(x.restaurant, requested_cuisine),  # Higher "best" boost first
                x.final_score,
                x.restaurant.rating,
                get_award_level(x.restaurant) if has_award(x.restaurant) else 0.0
            ),
            reverse=True
        )
        
        # Sort award winners / secondary cuisine by: award level (desc) + final_score (desc) + rating (desc)
        award_winners_secondary.sort(
            key=lambda x: (
                get_award_level(x.restaurant) if has_award(x.restaurant) else 0.0,
                x.final_score,
                x.restaurant.rating
            ),
            reverse=True
        )
        
        # Sort others by: rating (desc) + final_score (desc)
        others.sort(
            key=lambda x: (x.restaurant.rating, x.final_score),
            reverse=True
        )
        
        logger.info(f"RANKING SUMMARY: {len(primary_cuisine)} primary cuisine, {len(award_winners_secondary)} award/secondary, {len(others)} others")
        if primary_cuisine:
            logger.info(f"  Primary cuisine: {[r.restaurant.name for r in primary_cuisine]}")
        if award_winners_secondary:
            logger.info(f"  Award/Secondary: {[r.restaurant.name for r in award_winners_secondary]}")
        
        # Combine: primary cuisine first, then award/secondary, then others
        re_ranked_top_n = primary_cuisine + award_winners_secondary + others
        
        # Return re-ranked top N + rest
        return re_ranked_top_n + rest


class AdvancedScoreFusion(ScoreFusion):
    """Advanced score fusion with non-linear transformations and interaction terms."""
    
    def fuse(
        self,
        restaurant: Restaurant,
        vibe_score: float,
        cuisine_score: float,
        price_score: float,
        feature_score: float,
        weights: SignalWeights,
        parsed_query: ParsedQuery | None = None,
    ) -> ScoredRestaurant:
        """Compute advanced fusion of all scores."""
        if parsed_query is None:
            raise ValueError("AdvancedScoreFusion requires parsed_query parameter")
        
        transformed_scores = {
            'vibe': self._transform_signal(vibe_score, 'vibe'),
            'cuisine': self._transform_signal(cuisine_score, 'cuisine'),
            'price': self._transform_signal(price_score, 'price'),
            'features': self._transform_signal(feature_score, 'features'),
        }
        
        interaction_bonus = self._compute_interaction_bonus(
            transformed_scores, weights, parsed_query
        )
        
        perfect_match_boost = self._compute_perfect_match_boost(
            transformed_scores, parsed_query
        )
        
        penalty = self._compute_penalty(
            transformed_scores, weights, parsed_query, restaurant
        )
        
        base_score = (
            weights.vibe * transformed_scores['vibe'] +
            weights.cuisine * transformed_scores['cuisine'] +
            weights.price * transformed_scores['price'] +
            weights.features * transformed_scores['features']
        )
        
        # Log detailed scoring for top candidates (to debug ranking issues)
        # Log if this is a high-scoring restaurant or if there's a cuisine mismatch
        # Calculate preliminary final score for logging decision (without award boost yet)
        preliminary_score = base_score + interaction_bonus + perfect_match_boost - penalty
        should_log = (
            preliminary_score > 0.8 or  # High scoring
            (parsed_query.preferences.cuisine and cuisine_score < 0.3 and preliminary_score > 0.7) 
        )
        
        # Award boost: Prioritize award-winning restaurants
        award_boost = 0.0
        if has_award(restaurant):
            award_level = get_award_level(restaurant)
            # Award boost scales with award level (Gold = 0.15, Silver = 0.12, Honorable = 0.08, Other = 0.05)
            award_boost = 0.15 * award_level
            if should_log:
                logger.debug(f"   Award Boost: {award_boost:.4f} (level: {award_level:.2f})")
        
        # Primary cuisine boost: Prioritize restaurants where requested cuisine is PRIMARY
        # This gives a significant boost to restaurants that primarily serve the requested cuisine
        primary_cuisine_boost = 0.0
        if parsed_query and parsed_query.preferences.cuisine and cuisine_score >= 0.95:
            # Only apply boost if cuisine score is very high (0.95+) indicating it's a primary match
            # cuisine_score of 0.75 means secondary cuisine, so no boost
            # cuisine_score of 1.0 means primary cuisine, so we boost it
            primary_cuisine_boost = 0.12  # Strong boost for primary cuisine restaurants
            if should_log:
                logger.debug(f"   Primary Cuisine Boost: {primary_cuisine_boost:.4f} (cuisine_score={cuisine_score:.2f} indicates primary match)")
        
        final_score = base_score + interaction_bonus + perfect_match_boost + award_boost + primary_cuisine_boost - penalty
        final_score = max(0.0, min(1.0, final_score))
        
        if should_log:
            logger.debug(f"\n📊 Detailed Scoring: {restaurant.name}")
            logger.debug(f"   Raw Scores: vibe={vibe_score:.3f}, cuisine={cuisine_score:.3f}, price={price_score:.3f}, feature={feature_score:.3f}")
            logger.debug(f"   Transformed: vibe={transformed_scores['vibe']:.3f}, cuisine={transformed_scores['cuisine']:.3f}")
            logger.debug(f"   Weights: vibe={weights.vibe:.2f}, cuisine={weights.cuisine:.2f}, price={weights.price:.2f}, feature={weights.features:.2f}")
            logger.debug(f"   Base Score: {base_score:.4f}")
            logger.debug(f"   Interaction Bonus: {interaction_bonus:.4f}")
            logger.debug(f"   Perfect Match Boost: {perfect_match_boost:.4f}")
            logger.debug(f"   Penalty: {penalty:.4f}")
            logger.debug(f"   Final Score: {final_score:.4f}")
        
        # Ensure all scores are in valid range [0, 1] to satisfy Pydantic schema
        clamped_vibe_score = max(0.0, min(1.0, vibe_score))
        clamped_cuisine_score = max(0.0, min(1.0, cuisine_score))
        clamped_price_score = max(0.0, min(1.0, price_score))
        clamped_feature_score = max(0.0, min(1.0, feature_score))
        
        return ScoredRestaurant(
            restaurant=restaurant,
            vibe_score=clamped_vibe_score,
            cuisine_score=clamped_cuisine_score,
            price_score=clamped_price_score,
            feature_score=clamped_feature_score,
            final_score=final_score,
        )
    
    def _transform_signal(self, score: float, signal_type: str) -> float:
        """Apply non-linear transformation to signal score."""
        if score <= 0:
            return 0.0
        if score >= 1.0:
            return 1.0
        
        transformed = math.sqrt(score)
        
        if score >= 0.95:
            boost = 0.05 * (score - 0.95) / 0.05
            transformed = min(1.0, transformed + boost)
        
        return transformed
    
    def _compute_interaction_bonus(
        self,
        scores: dict[str, float],
        weights: SignalWeights,
        parsed_query: ParsedQuery,
    ) -> float:
        """Compute bonus when multiple signals align well."""
        bonus = 0.0

        # BONUS FIX: Lower threshold from 0.7 to 0.6 to catch near-perfect matches
        # Elderly users with exact needs (Italian + outdoor seating) often score 0.6-0.7 on both
        if scores['cuisine'] > 0.6 and scores['features'] > 0.6:
            interaction_strength = (scores['cuisine'] * scores['features'])
            bonus += 0.1 * interaction_strength * weights.cuisine * weights.features

        if scores['vibe'] > 0.6 and scores['cuisine'] > 0.6:
            interaction_strength = (scores['vibe'] * scores['cuisine'])
            bonus += 0.08 * interaction_strength * weights.vibe * weights.cuisine

        if scores['vibe'] > 0.6 and scores['features'] > 0.6:
            interaction_strength = (scores['vibe'] * scores['features'])
            bonus += 0.08 * interaction_strength * weights.vibe * weights.features

        if (scores['cuisine'] > 0.6 and scores['vibe'] > 0.6 and scores['features'] > 0.6):
            triple_strength = scores['cuisine'] * scores['vibe'] * scores['features']
            bonus += 0.12 * triple_strength

        return min(0.2, bonus)
    
    def _compute_perfect_match_boost(
        self,
        scores: dict[str, float],
        parsed_query: ParsedQuery,
    ) -> float:
        """Apply exponential boost for perfect or near-perfect matches."""
        boost = 0.0
        
        if scores['cuisine'] >= 0.95 and parsed_query.preferences.cuisine:
            excess = scores['cuisine'] - 0.95
            boost += 0.08 * (excess / 0.05) ** 2
        
        if scores['features'] >= 0.95 and parsed_query.preferences.features:
            excess = scores['features'] - 0.95
            boost += 0.08 * (excess / 0.05) ** 2
        
        if scores['vibe'] >= 0.95:
            excess = scores['vibe'] - 0.95
            boost += 0.05 * (excess / 0.05) ** 2
        
        return min(0.15, boost)
    
    def _compute_penalty(
        self,
        scores: dict[str, float],
        weights: SignalWeights,
        parsed_query: ParsedQuery,
        restaurant: Restaurant,
    ) -> float:
        """Apply penalties for missing critical requirements."""
        penalty = 0.0

        # PRIORITY 1 FIX: Accessibility-specific hard penalties (SAFETY CRITICAL)
        # Elderly users with mobility needs must get accurate recommendations
        query_lower = parsed_query.raw_query.lower()

        ACCESSIBILITY_KEYWORDS = [
            'wheelchair', 'accessible', 'ramp', 'walker', 'mobility',
            'elevator', 'handicap', 'disabled', 'disability',
            'parking', 'park', 'senior', 'elderly', 'cane', 'crutch'
        ]

        QUIET_KEYWORDS = ['quiet', 'not loud', 'not noisy', 'peaceful', 'calm', 'tranquil']

        # Check if query has accessibility needs
        has_accessibility_need = any(keyword in query_lower for keyword in ACCESSIBILITY_KEYWORDS)
        has_quiet_need = any(keyword in query_lower for keyword in QUIET_KEYWORDS)

        # CRITICAL: Hard penalty for missing accessibility features (safety issue)
        if has_accessibility_need and scores['features'] < 0.5:
            # Missing accessibility is DANGEROUS - apply strong penalty
            penalty += 0.5 * (0.5 - scores['features'])  # Up to 0.25 penalty

        # CRITICAL: Hard penalty for loud restaurants when quiet is requested
        if has_quiet_need:
            # Check if restaurant is loud
            if restaurant.vibe.noise_level == "loud":
                penalty += 0.4  # Strong penalty for loud when quiet requested
            elif scores['features'] < 0.4:
                penalty += 0.2  # Penalty if quiet feature poorly matched

        if weights.cuisine > 0.3 and scores['cuisine'] < 0.3:
            penalty += 0.15 * weights.cuisine * (0.3 - scores['cuisine']) / 0.3

        if weights.features > 0.4 and scores['features'] < 0.3:
            penalty += 0.15 * weights.features * (0.3 - scores['features']) / 0.3

        if weights.vibe > 0.4 and scores['vibe'] < 0.3:
            penalty += 0.12 * weights.vibe * (0.3 - scores['vibe']) / 0.3

        if parsed_query.preferences.features and scores['features'] < 0.1:
            penalty += 0.2 * weights.features

        # PRIORITY 2 FIX: Graduated cuisine penalty (less harsh for close matches)
        if parsed_query.preferences.cuisine and scores['cuisine'] < 0.5:
            # Scale penalty: 0.45 score = small penalty, 0.1 score = large penalty
            mismatch_severity = 0.5 - scores['cuisine']
            cuisine_penalty = 0.3 * (mismatch_severity / 0.4)  # Max 0.3 penalty
            penalty += cuisine_penalty
            
            # Log cuisine mismatch penalties for debugging
            if restaurant.name == "MORIMOTO MAUI" or penalty > 0.2:
                logger.debug(f"   ⚠️  Cuisine Mismatch Penalty: {cuisine_penalty:.4f} (cuisine_score={scores['cuisine']:.3f}, requested={parsed_query.preferences.cuisine}, restaurant={restaurant.cuisine})")

        # EXPANDED: More exclusivity keywords for elderly users
        has_exclusivity_keywords = any(
            word in query_lower for word in [
                'pure', 'only', 'exclusively', 'strictly',
                'just', 'real', 'authentic', 'genuine', 'traditional',
                'must be', 'has to be', 'needs to be',
                'no fusion', 'not fusion'
            ]
        )

        if has_exclusivity_keywords:
            if 'vegan' in query_lower and 'vegan' not in restaurant.cuisine.lower():
                penalty += 0.25

            if 'vegetarian' in query_lower and 'vegetarian' not in restaurant.cuisine.lower():
                if not restaurant.features.get('serves_vegetarian', False):
                    penalty += 0.2

        # Increased max penalty for accessibility safety
        return min(0.5, penalty)  # Was 0.3, now 0.5 for accessibility penalties

