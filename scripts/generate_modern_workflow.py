"""
Generate a modern, Figma-style workflow diagram for Ana AI system.
Clean, professional design with proper spacing and no overlapping elements.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import matplotlib.lines as mlines

# Set up modern, clean style
plt.style.use('seaborn-v0_8-darkgrid')
fig = plt.figure(figsize=(24, 32), facecolor='white')
ax = fig.add_subplot(111)
ax.set_xlim(0, 24)
ax.set_ylim(0, 32)
ax.axis('off')
fig.patch.set_facecolor('white')

# Modern color palette - Figma-inspired
COLORS = {
    'primary': '#4F46E5',      # Indigo (primary actions)
    'secondary': '#06B6D4',    # Cyan (secondary elements)
    'success': '#10B981',      # Green (data/success)
    'warning': '#F59E0B',      # Amber (scoring)
    'danger': '#EF4444',       # Red (LLM/critical)
    'purple': '#8B5CF6',       # Purple (fusion)
    'gray': '#6B7280',         # Gray (neutral)
    'light': '#F3F4F6',        # Light gray (backgrounds)
    'dark': '#1F2937',         # Dark gray (text)
}

def create_modern_box(ax, x, y, width, height, text, bg_color,
                     border_color=None, text_color='white',
                     fontsize=11, bold=False, alpha=1.0):
    """Create a modern rounded box with shadow effect."""
    # Shadow
    shadow = FancyBboxPatch(
        (x + 0.05, y - 0.05), width, height,
        boxstyle="round,pad=0.15",
        edgecolor='none',
        facecolor='black',
        alpha=0.1,
        zorder=1
    )
    ax.add_patch(shadow)

    # Main box
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.15",
        edgecolor=border_color or bg_color,
        facecolor=bg_color,
        linewidth=2,
        alpha=alpha,
        zorder=2
    )
    ax.add_patch(box)

    # Text
    weight = 'bold' if bold else 'normal'
    ax.text(
        x + width/2, y + height/2, text,
        ha='center', va='center',
        fontsize=fontsize, color=text_color,
        weight=weight, zorder=3,
        wrap=True,
        family='sans-serif'
    )
    return box

def create_modern_arrow(ax, x1, y1, x2, y2, color=COLORS['gray'],
                       width=2.5, style='solid'):
    """Create a modern arrow with smooth curves."""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='->,head_width=0.5,head_length=0.5',
        color=color,
        linewidth=width,
        linestyle=style,
        zorder=1,
        alpha=0.8
    )
    ax.add_patch(arrow)
    return arrow

def create_phase_header(ax, x, y, width, text, number):
    """Create a modern phase header with number badge."""
    # Phase bar
    create_modern_box(ax, x, y, width, 0.9, '', COLORS['primary'], alpha=0.1)

    # Number badge
    create_modern_box(ax, x + 0.3, y + 0.15, 0.6, 0.6, str(number),
                     COLORS['primary'], fontsize=14, bold=True)

    # Phase text
    ax.text(x + 1.2, y + 0.45, text, ha='left', va='center',
            fontsize=14, weight='bold', color=COLORS['dark'],
            family='sans-serif')

# ============================================================================
# HEADER
# ============================================================================
y_current = 31

# Title
ax.text(12, y_current, 'Ana AI - Complete Workflow',
        ha='center', va='top', fontsize=28, weight='bold',
        color=COLORS['primary'], family='sans-serif')

y_current -= 1
ax.text(12, y_current, 'Vibe-Based Restaurant Search System • Hybrid AI Architecture',
        ha='center', va='top', fontsize=13, style='italic',
        color=COLORS['gray'], family='sans-serif')

# User query box
y_current -= 1.5
create_modern_box(ax, 6, y_current, 12, 1,
                 'USER QUERY\n"romantic hawaiian spot with ocean views"',
                 COLORS['danger'], text_color='white', fontsize=13, bold=True)

create_modern_arrow(ax, 12, y_current, 12, y_current - 1, COLORS['primary'], 3)

# ============================================================================
# PHASE 1: QUERY UNDERSTANDING
# ============================================================================
y_current -= 2
create_phase_header(ax, 1, y_current, 22, 'QUERY UNDERSTANDING', 1)

y_current -= 1.8
# Three main processes
box_width = 6.5
spacing = 0.5

create_modern_box(ax, 1.5, y_current, box_width, 1.5,
                 'Typo Correction\n\nhawaain→hawaiian\nindain→indian',
                 COLORS['secondary'], fontsize=10)

create_modern_box(ax, 1.5 + box_width + spacing, y_current, box_width, 1.5,
                 'Restaurant Name Detection\n\nFuzzy matching\n"mama fish" → "Mama\'s Fish"',
                 COLORS['secondary'], fontsize=10)

create_modern_box(ax, 1.5 + 2*(box_width + spacing), y_current, box_width, 1.5,
                 'LLM Parsing\nGemini Flash\n\nExtract structured intent',
                 COLORS['danger'], fontsize=10, bold=True)

# Arrows down to enrichment
y_enrich = y_current - 2
create_modern_arrow(ax, 4.75, y_current, 4.75, y_enrich + 1)
create_modern_arrow(ax, 12, y_current, 12, y_enrich + 1)
create_modern_arrow(ax, 19.25, y_current, 19.25, y_enrich + 1)

# Fallback enrichment
create_modern_box(ax, 3, y_enrich, 18, 1,
                 'Fallback Pattern Enrichment: Extract cuisine, features, location from patterns',
                 COLORS['secondary'], fontsize=10, alpha=0.8)

# ParsedQuery output
y_parsed = y_enrich - 2.5
create_modern_arrow(ax, 12, y_enrich, 12, y_parsed + 2.3)

create_modern_box(ax, 3, y_parsed, 18, 2.3,
                 'ParsedQuery Object (Structured Intent)\n\n' +
                 'semantic_query: "romantic restaurant hawaiian cuisine ocean views..."\n' +
                 'must_not: {formality[], price[], cuisine[]}  |  preferences: {cuisine["Hawaiian"], atmosphere["romantic"]}\n' +
                 'weights: {vibe:0.6, cuisine:0.2, price:0.1, features:0.1}',
                 COLORS['success'], text_color='white', fontsize=9, bold=True)

# ============================================================================
# PHASE 2: CANDIDATE RETRIEVAL & FILTERING
# ============================================================================
y_current = y_parsed - 1.5
create_phase_header(ax, 1, y_current, 22, 'CANDIDATE RETRIEVAL & FILTERING', 2)
create_modern_arrow(ax, 12, y_parsed, 12, y_current + 0.9, COLORS['primary'], 3)

y_current -= 1.8
# Three processes
create_modern_box(ax, 1.5, y_current, box_width, 1.5,
                 'Exact Match Check\n\nIf found:\n→ 0.98 vibe boost\n→ Guaranteed #1',
                 COLORS['secondary'], fontsize=9)

create_modern_box(ax, 1.5 + box_width + spacing, y_current, box_width, 1.5,
                 'Dynamic n_results\n\nBase: 20\n+10 if cuisine explicit\n+10 if features ≥ 2',
                 COLORS['secondary'], fontsize=9)

create_modern_box(ax, 1.5 + 2*(box_width + spacing), y_current, box_width, 1.5,
                 'Cuisine Enhancement\n\nRepeat cuisine 3x\nfor better embeddings',
                 COLORS['secondary'], fontsize=9)

# Vector search
y_vector = y_current - 2
create_modern_arrow(ax, 4.75, y_current, 4.75, y_vector + 1.5)
create_modern_arrow(ax, 12, y_current, 12, y_vector + 1.5)
create_modern_arrow(ax, 19.25, y_current, 19.25, y_vector + 1.5)

create_modern_box(ax, 4, y_vector, 16, 1.5,
                 'VECTOR SEARCH (ChromaDB)\n\n' +
                 'Model: text-embedding-004  |  Method: Cosine similarity  |  Returns: Top 20-40 from 360 restaurants',
                 COLORS['danger'], fontsize=10, bold=True)

# Hard filters
y_filter = y_vector - 2
create_modern_arrow(ax, 12, y_vector, 12, y_filter + 1)

filter_width = 4.2
filter_spacing = 0.2
create_modern_box(ax, 1.8, y_filter, filter_width, 1,
                 'Business\nStatus', COLORS['secondary'], fontsize=9)
create_modern_box(ax, 1.8 + filter_width + filter_spacing, y_filter, filter_width, 1,
                 'Formality\nFilter', COLORS['secondary'], fontsize=9)
create_modern_box(ax, 1.8 + 2*(filter_width + filter_spacing), y_filter, filter_width, 1,
                 'Cuisine\n(if explicit)', COLORS['secondary'], fontsize=9, bold=True)
create_modern_box(ax, 1.8 + 3*(filter_width + filter_spacing), y_filter, filter_width, 1,
                 'Features\nFilter', COLORS['secondary'], fontsize=9)
create_modern_box(ax, 1.8 + 4*(filter_width + filter_spacing), y_filter, filter_width, 1,
                 'Location\nFilter', COLORS['secondary'], fontsize=9)

# Filtered candidates
y_candidates = y_filter - 1.5
create_modern_arrow(ax, 12, y_filter, 12, y_candidates + 0.8, COLORS['primary'], 3)

create_modern_box(ax, 5, y_candidates, 14, 0.8,
                 'FILTERED CANDIDATES: 10-30 restaurants ready for scoring',
                 COLORS['success'], fontsize=11, bold=True)

# ============================================================================
# PHASE 3: MULTI-SIGNAL SCORING
# ============================================================================
y_current = y_candidates - 1.5
create_phase_header(ax, 1, y_current, 22, 'MULTI-SIGNAL SCORING (Parallel Execution)', 3)
create_modern_arrow(ax, 12, y_candidates, 12, y_current + 0.9, COLORS['primary'], 3)

y_current -= 3
# Four scorers
scorer_width = 5.2
scorer_spacing = 0.3

create_modern_box(ax, 1.5, y_current, scorer_width, 2.5,
                 'VIBE SCORER\n\n' +
                 'Method: Cosine\nsimilarity\n\n' +
                 'Input: Query & Vibe\nembeddings\n\n' +
                 'Score: 0.925',
                 COLORS['warning'], fontsize=9, bold=True)

create_modern_box(ax, 1.5 + scorer_width + scorer_spacing, y_current, scorer_width, 2.5,
                 'CUISINE SCORER\n\n' +
                 'Method: Fuzzy\nstring matching\n\n' +
                 'Input: Requested vs\nRestaurant cuisine\n\n' +
                 'Score: 1.0',
                 COLORS['warning'], fontsize=9, bold=True)

create_modern_box(ax, 1.5 + 2*(scorer_width + scorer_spacing), y_current, scorer_width, 2.5,
                 'PRICE SCORER\n\n' +
                 'Method: Euclidean\ndistance\n\n' +
                 'Input: Price level\ncomparison\n\n' +
                 'Score: 0.75',
                 COLORS['warning'], fontsize=9, bold=True)

create_modern_box(ax, 1.5 + 3*(scorer_width + scorer_spacing), y_current, scorer_width, 2.5,
                 'FEATURE SCORER\n\n' +
                 'Method: Jaccard\nsimilarity\n\n' +
                 'Input: Feature sets\nintersection\n\n' +
                 'Score: 0.65',
                 COLORS['warning'], fontsize=9, bold=True)

# Async gather note
create_modern_box(ax, 8, y_current - 0.9, 8, 0.7,
                 'async gather() - All scorers run in parallel',
                 COLORS['primary'], alpha=0.2, text_color=COLORS['dark'], fontsize=10, bold=True)

# Scored output
y_scored = y_current - 2
create_modern_arrow(ax, 4, y_current, 4, y_scored + 0.9)
create_modern_arrow(ax, 9.3, y_current, 9.3, y_scored + 0.9)
create_modern_arrow(ax, 14.6, y_current, 14.6, y_scored + 0.9)
create_modern_arrow(ax, 19.9, y_current, 19.9, y_scored + 0.9)

create_modern_box(ax, 4, y_scored, 16, 0.9,
                 'Each Restaurant: vibe=0.925  |  cuisine=1.0  |  price=0.75  |  features=0.65',
                 COLORS['success'], fontsize=10, bold=True)

# ============================================================================
# PHASE 4: ADVANCED SCORE FUSION
# ============================================================================
y_current = y_scored - 1.5
create_phase_header(ax, 1, y_current, 22, 'ADVANCED SCORE FUSION', 4)
create_modern_arrow(ax, 12, y_scored, 12, y_current + 0.9, COLORS['primary'], 3)

y_current -= 2
# Fusion components
fusion_width = 3.8
fusion_spacing = 0.25

create_modern_box(ax, 2, y_current, fusion_width, 1.5,
                 'Transform\n\nsqrt(score)\nboost high scores',
                 COLORS['purple'], fontsize=9)

create_modern_box(ax, 2 + fusion_width + fusion_spacing, y_current, fusion_width, 1.5,
                 'Weighted Sum\n\n0.6×0.925 +\n0.2×1.0 = 0.895',
                 COLORS['purple'], fontsize=9, bold=True)

create_modern_box(ax, 2 + 2*(fusion_width + fusion_spacing), y_current, fusion_width, 1.5,
                 'Interaction\nBonuses\n\nMulti-signal\nalignment',
                 COLORS['purple'], fontsize=9)

create_modern_box(ax, 2 + 3*(fusion_width + fusion_spacing), y_current, fusion_width, 1.5,
                 'Perfect Match\nBoosts\n\nExponential\nfor ≥0.95',
                 COLORS['purple'], fontsize=9)

create_modern_box(ax, 2 + 4*(fusion_width + fusion_spacing), y_current, fusion_width, 1.5,
                 'Award Boost\n\nGold: +0.15\nSilver: +0.12',
                 COLORS['purple'], fontsize=9)

# Penalties
y_penalty = y_current - 2
for i in range(5):
    x_pos = 2 + i * (fusion_width + fusion_spacing) + fusion_width/2
    create_modern_arrow(ax, x_pos, y_current, x_pos, y_penalty + 1.3)

create_modern_box(ax, 4.5, y_penalty, 15, 1.3,
                 'PENALTIES (Safety-Critical)\n\n' +
                 'Accessibility: -0.25  |  Loud when quiet: -0.40  |  Cuisine mismatch: -0.30',
                 COLORS['danger'], fontsize=10, bold=True)

# Final score
y_final = y_penalty - 1.5
create_modern_arrow(ax, 12, y_penalty, 12, y_final + 0.9, COLORS['primary'], 3)

create_modern_box(ax, 5.5, y_final, 13, 0.9,
                 'FINAL SCORE = base + bonuses + awards - penalties = 0.89',
                 COLORS['success'], fontsize=11, bold=True)

# ============================================================================
# PHASE 5: RANKING
# ============================================================================
y_current = y_final - 1.5
create_phase_header(ax, 1, y_current, 22, 'RANKING WITH AWARD PRIORITY', 5)
create_modern_arrow(ax, 12, y_final, 12, y_current + 0.9, COLORS['primary'], 3)

y_current -= 2
create_modern_box(ax, 3, y_current, 8, 1.5,
                 'Initial Sort\n\nSort all by final_score (desc)',
                 COLORS['secondary'], fontsize=10)

create_modern_box(ax, 13, y_current, 8, 1.5,
                 'Award Priority (Top 10)\n\nAward winners first by:\nlevel → rating → score',
                 COLORS['secondary'], fontsize=10, bold=True)

# Ranked results
y_ranked = y_current - 1.8
create_modern_arrow(ax, 7, y_current, 7, y_ranked + 1)
create_modern_arrow(ax, 17, y_current, 17, y_ranked + 1)
create_modern_arrow(ax, 12, y_current, 12, y_ranked + 1, COLORS['primary'], 3)

create_modern_box(ax, 4, y_ranked, 16, 1,
                 'RANKED RESULTS\n' +
                 '#1: Mama\'s Fish House (0.89)  |  #2: Merriman\'s Kapalua (0.85)  |  #3: Lahaina Grill (0.82)',
                 COLORS['success'], fontsize=10, bold=True)

# ============================================================================
# PHASE 6: RESPONSE GENERATION
# ============================================================================
y_current = y_ranked - 1.5
create_phase_header(ax, 1, y_current, 22, 'RESPONSE GENERATION', 6)
create_modern_arrow(ax, 12, y_ranked, 12, y_current + 0.9, COLORS['primary'], 3)

y_current -= 2
# Response components
resp_width = 4.2
resp_spacing = 0.2

create_modern_box(ax, 1.8, y_current, resp_width, 1.5,
                 'Select\nTop + Alts\n\n3-9 results',
                 COLORS['secondary'], fontsize=9)

create_modern_box(ax, 1.8 + resp_width + resp_spacing, y_current, resp_width, 1.5,
                 'Generate\nMatch Reasons\n\nWhy matched',
                 COLORS['secondary'], fontsize=9)

create_modern_box(ax, 1.8 + 2*(resp_width + resp_spacing), y_current, resp_width, 1.5,
                 'Lookup\nPhotos &\nVideos',
                 COLORS['secondary'], fontsize=9)

create_modern_box(ax, 1.8 + 3*(resp_width + resp_spacing), y_current, resp_width, 1.5,
                 'Build\nPrompt\n\nCombine data',
                 COLORS['secondary'], fontsize=9)

create_modern_box(ax, 1.8 + 4*(resp_width + resp_spacing), y_current, resp_width, 1.5,
                 'LLM Generate\nGemini Flash\n\nTemp: 0.4',
                 COLORS['danger'], fontsize=9, bold=True)

# Final response
y_response = y_current - 2
create_modern_arrow(ax, 12, y_current, 12, y_response + 1.3, COLORS['primary'], 3)

create_modern_box(ax, 3, y_response, 18, 1.3,
                 'FINAL ANA RESPONSE\n\n' +
                 'Top Match + Alternatives + Match Reasons + Natural Language Explanation + Photos & Videos + Confidence',
                 COLORS['success'], fontsize=10, bold=True)

# User output
y_user = y_response - 1.5
create_modern_arrow(ax, 12, y_response, 12, y_user + 0.9, COLORS['primary'], 3)

create_modern_box(ax, 4, y_user, 16, 0.9,
                 'USER: "Mama\'s Fish House is an excellent match... Check out photos: [URLs]"',
                 COLORS['primary'], fontsize=11, bold=True)

# ============================================================================
# FOOTER: KEY METRICS
# ============================================================================
y_footer = 1.5

# Key features
create_modern_box(ax, 1, y_footer, 11, 1.2,
                 'KEY FEATURES\n' +
                 'LLM Understanding • Hybrid Retrieval • 4 Parallel Scorers • Advanced Fusion • Award Recognition',
                 COLORS['light'], text_color=COLORS['dark'], fontsize=9, bold=True, alpha=1)

# Performance
create_modern_box(ax, 12.5, y_footer, 11, 1.2,
                 'PERFORMANCE METRICS\n' +
                 'Understanding: 400ms • Vector: 100ms • Scoring: 300ms • Fusion: 10ms • Ranking: 5ms • Response: 500ms • TOTAL: 1.3s',
                 COLORS['light'], text_color=COLORS['dark'], fontsize=9, bold=True, alpha=1)

# Branding
ax.text(12, 0.5, 'Ana AI • Powered by Google Gemini • Hybrid AI Architecture',
        ha='center', va='center', fontsize=10, style='italic',
        color=COLORS['gray'], alpha=0.7, family='sans-serif')

# Save
plt.tight_layout()
plt.savefig('/Users/yugandhargopu/Ana-Assessment/docs/workflow_diagram.png',
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print("✅ Modern workflow diagram saved!")
print("   Location: docs/workflow_diagram.png")
print("   Resolution: 300 DPI (presentation-quality)")
print("   Style: Figma-inspired modern design")
