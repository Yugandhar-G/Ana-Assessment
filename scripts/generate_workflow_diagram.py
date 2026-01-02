"""
Generate a professional workflow diagram for Ana AI system.
Outputs a PNG file suitable for presentations.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.lines as mlines

# Set up the figure with a clean, professional style
fig, ax = plt.subplots(figsize=(20, 28))
ax.set_xlim(0, 20)
ax.set_ylim(0, 28)
ax.axis('off')

# Color scheme - professional and clean
COLOR_PHASE = '#2C3E50'  # Dark blue-gray for phase headers
COLOR_PROCESS = '#3498DB'  # Blue for process boxes
COLOR_DATA = '#27AE60'  # Green for data/output boxes
COLOR_LLM = '#E74C3C'  # Red for LLM operations
COLOR_SCORING = '#F39C12'  # Orange for scoring
COLOR_FUSION = '#9B59B6'  # Purple for fusion
COLOR_TEXT = '#2C3E50'  # Dark text
COLOR_ARROW = '#7F8C8D'  # Gray arrows
COLOR_HIGHLIGHT = '#E67E22'  # Orange for highlights

# Helper function to create rounded rectangles
def create_box(ax, x, y, width, height, text, color, text_color='white', fontsize=10, bold=False):
    """Create a rounded rectangle box with text."""
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.1",
        edgecolor='black',
        facecolor=color,
        linewidth=2,
        zorder=2
    )
    ax.add_patch(box)

    # Add text
    weight = 'bold' if bold else 'normal'
    ax.text(
        x + width/2, y + height/2, text,
        ha='center', va='center',
        fontsize=fontsize, color=text_color,
        weight=weight, zorder=3,
        wrap=True
    )
    return box

def create_arrow(ax, x1, y1, x2, y2, color=COLOR_ARROW, width=2):
    """Create an arrow between two points."""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='->,head_width=0.4,head_length=0.4',
        color=color,
        linewidth=width,
        zorder=1
    )
    ax.add_patch(arrow)
    return arrow

# Title
ax.text(10, 27, 'Ana AI - Complete Workflow',
        ha='center', va='center', fontsize=24, weight='bold', color=COLOR_PHASE)
ax.text(10, 26.3, 'Vibe-Based Restaurant Search System',
        ha='center', va='center', fontsize=14, style='italic', color=COLOR_TEXT)

# USER QUERY
create_box(ax, 6, 25, 8, 0.8, 'USER QUERY: "romantic hawaiian spot"', COLOR_HIGHLIGHT, fontsize=12, bold=True)
create_arrow(ax, 10, 25, 10, 24.2)

# ============================================================================
# PHASE 1: QUERY UNDERSTANDING
# ============================================================================
y_phase1 = 23.5
create_box(ax, 1, y_phase1, 18, 0.7, 'PHASE 1: QUERY UNDERSTANDING (QueryParser)', COLOR_PHASE, fontsize=12, bold=True)

# Sub-processes
y_sub = y_phase1 - 1.2
create_box(ax, 1.5, y_sub, 5, 1.2, 'Typo Correction\n\nhawaain→hawaiian\nindain→indian\nnormalize text', COLOR_PROCESS, fontsize=9)
create_box(ax, 7.5, y_sub, 5, 1.2, 'Restaurant Name\nDetection\n\nFuzzy matching\n"mama fish" →\n"Mama\'s Fish House"', COLOR_PROCESS, fontsize=9)
create_box(ax, 13.5, y_sub, 5, 1.2, 'LLM Parsing\n(Gemini Flash)\n\nExtract structured\nintent as JSON', COLOR_LLM, fontsize=9)

# Arrows from phase to sub-processes
create_arrow(ax, 4, y_phase1 - 0.35, 4, y_sub + 1.2)
create_arrow(ax, 10, y_phase1 - 0.35, 10, y_sub + 1.2)
create_arrow(ax, 16, y_phase1 - 0.35, 16, y_sub + 1.2)

# Fallback enrichment
y_fallback = y_sub - 1.5
create_box(ax, 3, y_fallback, 14, 0.9, 'Fallback Pattern Enrichment: Extract cuisine, features, location from patterns', COLOR_PROCESS, fontsize=9)
create_arrow(ax, 10, y_sub, 10, y_fallback + 0.9)

# ParsedQuery output
y_parsed = y_fallback - 2.5
create_box(ax, 2, y_parsed, 16, 2,
           'ParsedQuery Object (Structured Intent)\n\n' +
           'semantic_query: "romantic restaurant hawaiian cuisine..."\n' +
           'must_not: {formality:[], price:[], cuisine:[]}\n' +
           'preferences: {cuisine:["Hawaiian"], atmosphere:["romantic"]}\n' +
           'weights: {vibe:0.6, cuisine:0.2, price:0.1, features:0.1}',
           COLOR_DATA, fontsize=9)
create_arrow(ax, 10, y_fallback, 10, y_parsed + 2)

# ============================================================================
# PHASE 2: CANDIDATE RETRIEVAL & FILTERING
# ============================================================================
y_phase2 = y_parsed - 1.2
create_box(ax, 1, y_phase2, 18, 0.7, 'PHASE 2: CANDIDATE RETRIEVAL & FILTERING', COLOR_PHASE, fontsize=12, bold=True)
create_arrow(ax, 10, y_parsed, 10, y_phase2 + 0.7)

# Sub-processes
y_sub2 = y_phase2 - 1.2
create_box(ax, 1.5, y_sub2, 5, 1.2, 'Exact Match\nCheck\n\nIf found:\n0.98 vibe boost\nGuaranteed #1', COLOR_PROCESS, fontsize=9)
create_box(ax, 7.5, y_sub2, 5, 1.2, 'Dynamic\nn_results\n\nBase: 20\n+10 if cuisine\n+10 if features≥2', COLOR_PROCESS, fontsize=9)
create_box(ax, 13.5, y_sub2, 5, 1.2, 'Cuisine Query\nEnhancement\n\nRepeat cuisine 3x\nfor better embeddings', COLOR_PROCESS, fontsize=9)

create_arrow(ax, 4, y_phase2, 4, y_sub2 + 1.2)
create_arrow(ax, 10, y_phase2, 10, y_sub2 + 1.2)
create_arrow(ax, 16, y_phase2, 16, y_sub2 + 1.2)

# Vector search
y_vector = y_sub2 - 1.8
create_box(ax, 3, y_vector, 14, 1.3,
           'VECTOR SEARCH (ChromaDB)\n\n' +
           'Model: text-embedding-004 | Method: Cosine similarity\n' +
           'Returns: Top 20-40 candidates from 360 restaurants',
           COLOR_LLM, fontsize=9, bold=True)
create_arrow(ax, 10, y_sub2, 10, y_vector + 1.3)

# Hard filtering
y_filter = y_vector - 1.5
create_box(ax, 2, y_filter, 3.2, 0.9, 'Business\nStatus\nFilter', COLOR_PROCESS, fontsize=8)
create_box(ax, 5.5, y_filter, 3.2, 0.9, 'Formality\nFilter', COLOR_PROCESS, fontsize=8)
create_box(ax, 9, y_filter, 3.2, 0.9, 'Cuisine\nFilter\n(if explicit)', COLOR_PROCESS, fontsize=8)
create_box(ax, 12.5, y_filter, 3.2, 0.9, 'Features\nFilter', COLOR_PROCESS, fontsize=8)
create_box(ax, 16, y_filter, 3.2, 0.9, 'Location\nFilter', COLOR_PROCESS, fontsize=8)

create_arrow(ax, 10, y_vector, 10, y_filter + 0.9)

# Filtered candidates
y_candidates = y_filter - 1.3
create_box(ax, 4, y_candidates, 12, 0.8, 'FILTERED CANDIDATES (10-30 restaurants)', COLOR_DATA, fontsize=10, bold=True)
create_arrow(ax, 10, y_filter, 10, y_candidates + 0.8)

# ============================================================================
# PHASE 3: MULTI-SIGNAL SCORING (PARALLEL)
# ============================================================================
y_phase3 = y_candidates - 1
create_box(ax, 1, y_phase3, 18, 0.7, 'PHASE 3: MULTI-SIGNAL SCORING (PARALLEL EXECUTION)', COLOR_PHASE, fontsize=12, bold=True)
create_arrow(ax, 10, y_candidates, 10, y_phase3 + 0.7)

# Scoring boxes
y_score = y_phase3 - 2.5
create_box(ax, 1.5, y_score, 3.8, 2,
           'VIBE SCORER\n\n' +
           'Method:\nCosine\nsimilarity\n\n' +
           'Output:\n0.925',
           COLOR_SCORING, fontsize=9, bold=True)
create_box(ax, 5.8, y_score, 3.8, 2,
           'CUISINE SCORER\n\n' +
           'Method:\nFuzzy string\nmatching\n\n' +
           'Output:\n1.0',
           COLOR_SCORING, fontsize=9, bold=True)
create_box(ax, 10.1, y_score, 3.8, 2,
           'PRICE SCORER\n\n' +
           'Method:\nEuclidean\ndistance\n\n' +
           'Output:\n0.75',
           COLOR_SCORING, fontsize=9, bold=True)
create_box(ax, 14.4, y_score, 3.8, 2,
           'FEATURE SCORER\n\n' +
           'Method:\nJaccard\nsimilarity\n\n' +
           'Output:\n0.65',
           COLOR_SCORING, fontsize=9, bold=True)

# Arrows to scorers
create_arrow(ax, 3.4, y_phase3, 3.4, y_score + 2)
create_arrow(ax, 7.7, y_phase3, 7.7, y_score + 2)
create_arrow(ax, 12, y_phase3, 12, y_score + 2)
create_arrow(ax, 16.3, y_phase3, 16.3, y_score + 2)

# Add "async gather()" note
create_box(ax, 7.5, y_score - 0.7, 5, 0.5, 'async gather() - All run in parallel!', COLOR_HIGHLIGHT, fontsize=9)

# Scored output
y_scored = y_score - 1.8
create_box(ax, 3, y_scored, 14, 0.8,
           'Each Restaurant: vibe=0.925, cuisine=1.0, price=0.75, features=0.65',
           COLOR_DATA, fontsize=9, bold=True)
create_arrow(ax, 10, y_score, 10, y_scored + 0.8)

# ============================================================================
# PHASE 4: ADVANCED SCORE FUSION
# ============================================================================
y_phase4 = y_scored - 1
create_box(ax, 1, y_phase4, 18, 0.7, 'PHASE 4: ADVANCED SCORE FUSION', COLOR_PHASE, fontsize=12, bold=True)
create_arrow(ax, 10, y_scored, 10, y_phase4 + 0.7)

# Fusion components
y_fusion = y_phase4 - 1.5
create_box(ax, 1.5, y_fusion, 3.5, 1.2,
           'Transform\n\nsqrt(score)\nto boost\nhigh scores',
           COLOR_FUSION, fontsize=8)
create_box(ax, 5.3, y_fusion, 3.5, 1.2,
           'Weighted Sum\n\n0.6×0.925\n+0.2×1.0\n= 0.895',
           COLOR_FUSION, fontsize=8)
create_box(ax, 9.1, y_fusion, 3.5, 1.2,
           'Interaction\nBonuses\n\nMulti-signal\nalignment',
           COLOR_FUSION, fontsize=8)
create_box(ax, 12.9, y_fusion, 3.5, 1.2,
           'Perfect Match\nBoosts\n\nExponential\nfor ≥0.95',
           COLOR_FUSION, fontsize=8)
create_box(ax, 16.7, y_fusion, 2.8, 1.2,
           'Award\nBoost\n\nGold: +0.15',
           COLOR_FUSION, fontsize=8)

create_arrow(ax, 3.2, y_phase4, 3.2, y_fusion + 1.2)
create_arrow(ax, 7.0, y_phase4, 7.0, y_fusion + 1.2)
create_arrow(ax, 10.8, y_phase4, 10.8, y_fusion + 1.2)
create_arrow(ax, 14.6, y_phase4, 14.6, y_fusion + 1.2)
create_arrow(ax, 18.1, y_phase4, 18.1, y_fusion + 1.2)

# Penalties
y_penalty = y_fusion - 1.8
create_box(ax, 4, y_penalty, 12, 1.3,
           'PENALTIES (Safety-Critical)\n\n' +
           'Missing wheelchair: -0.25 | Loud when quiet: -0.40 | Cuisine mismatch: -0.30',
           COLOR_LLM, fontsize=9, bold=True)
create_arrow(ax, 10, y_fusion, 10, y_penalty + 1.3)

# Final score
y_final = y_penalty - 1.3
create_box(ax, 5, y_final, 10, 0.8,
           'FINAL SCORE = base + bonuses + award - penalties = 0.89',
           COLOR_DATA, fontsize=10, bold=True)
create_arrow(ax, 10, y_penalty, 10, y_final + 0.8)

# ============================================================================
# PHASE 5: RANKING WITH AWARD PRIORITY
# ============================================================================
y_phase5 = y_final - 1
create_box(ax, 1, y_phase5, 18, 0.7, 'PHASE 5: RANKING WITH AWARD PRIORITY', COLOR_PHASE, fontsize=12, bold=True)
create_arrow(ax, 10, y_final, 10, y_phase5 + 0.7)

# Ranking process
y_rank = y_phase5 - 1.5
create_box(ax, 2.5, y_rank, 7, 1.2,
           'Sort by Score\n\nAll restaurants\nsorted by final_score (desc)',
           COLOR_PROCESS, fontsize=9)
create_box(ax, 10.5, y_rank, 7, 1.2,
           'Award Priority (Top 10)\n\nAward winners first\nby level, rating, score',
           COLOR_PROCESS, fontsize=9)

create_arrow(ax, 6, y_phase5, 6, y_rank + 1.2)
create_arrow(ax, 14, y_phase5, 14, y_rank + 1.2)

# Ranked results
y_ranked = y_rank - 1.5
create_box(ax, 3, y_ranked, 14, 1,
           'RANKED RESULTS\n#1: Mama\'s Fish House (0.89) | #2: Merriman\'s Kapalua (0.85) | #3: Lahaina Grill (0.82)',
           COLOR_DATA, fontsize=9, bold=True)
create_arrow(ax, 10, y_rank, 10, y_ranked + 1)

# ============================================================================
# PHASE 6: RESPONSE GENERATION
# ============================================================================
y_phase6 = y_ranked - 1
create_box(ax, 1, y_phase6, 18, 0.7, 'PHASE 6: RESPONSE GENERATION', COLOR_PHASE, fontsize=12, bold=True)
create_arrow(ax, 10, y_ranked, 10, y_phase6 + 0.7)

# Response components
y_resp = y_phase6 - 1.2
create_box(ax, 1.5, y_resp, 3.5, 1,
           'Select\nTop + Alts\n\n3-9 results',
           COLOR_PROCESS, fontsize=8)
create_box(ax, 5.3, y_resp, 3.5, 1,
           'Generate\nMatch\nReasons',
           COLOR_PROCESS, fontsize=8)
create_box(ax, 9.1, y_resp, 3.5, 1,
           'Lookup\nPhotos &\nVideos',
           COLOR_PROCESS, fontsize=8)
create_box(ax, 12.9, y_resp, 2.8, 1,
           'Build\nPrompt',
           COLOR_PROCESS, fontsize=8)
create_box(ax, 16, y_resp, 3.5, 1,
           'LLM Generate\n(Gemini)\n\nTemp: 0.4',
           COLOR_LLM, fontsize=8)

create_arrow(ax, 3.2, y_phase6, 3.2, y_resp + 1)
create_arrow(ax, 7.0, y_phase6, 7.0, y_resp + 1)
create_arrow(ax, 10.8, y_phase6, 10.8, y_resp + 1)
create_arrow(ax, 14.3, y_phase6, 14.3, y_resp + 1)
create_arrow(ax, 17.7, y_phase6, 17.7, y_resp + 1)

# Final response
y_response = y_resp - 1.8
create_box(ax, 2, y_response, 16, 1.3,
           'FINAL ANA RESPONSE\n\n' +
           'Top Match + Alternatives + Match Reasons + Natural Language Explanation\n' +
           '+ Photos & Videos + Confidence Score',
           COLOR_DATA, fontsize=9, bold=True)
create_arrow(ax, 10, y_resp, 10, y_response + 1.3)

# User output
y_user = y_response - 1.3
create_box(ax, 4, y_user, 12, 0.8,
           'USER SEES: "Mama\'s Fish House is an excellent match... Check out photos: [URLs]"',
           COLOR_HIGHLIGHT, fontsize=10, bold=True)
create_arrow(ax, 10, y_response, 10, y_user + 0.8)

# ============================================================================
# SIDEBAR: KEY FEATURES & PERFORMANCE
# ============================================================================
# Add a legend/key features box at the bottom
y_legend = 1.5
create_box(ax, 0.5, y_legend, 9, 1.2,
           'KEY FEATURES:\n' +
           'LLM Query Understanding | Hybrid Retrieval | 4 Parallel Scorers\n' +
           'Advanced Fusion | Award Recognition | Natural Language Output',
           '#ECF0F1', text_color=COLOR_TEXT, fontsize=8)

create_box(ax, 10, y_legend, 9.5, 1.2,
           'PERFORMANCE:\n' +
           'Query Understanding: ~400ms | Vector Search: ~100ms | Scoring: ~300ms\n' +
           'Fusion: ~10ms | Ranking: ~5ms | Response Gen: ~500ms | TOTAL: ~1.3s',
           '#ECF0F1', text_color=COLOR_TEXT, fontsize=8)

# Add watermark/branding
ax.text(10, 0.5, 'Ana AI - Vibe-Based Restaurant Search System | Hybrid AI Architecture',
        ha='center', va='center', fontsize=10, style='italic', color=COLOR_TEXT, alpha=0.7)

# Save the figure
plt.tight_layout()
plt.savefig('/Users/yugandhargopu/Ana-Assessment/docs/workflow_diagram.png',
            dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Workflow diagram saved to: docs/workflow_diagram.png")
print("   Resolution: 300 DPI (presentation-quality)")
print("   Format: PNG")
