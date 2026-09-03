# WebAsset

A general-purpose Python CLI for analyzing internal link opportunities across real-world websites. Handles legacy CMS structures, duplicated content, boilerplate, tracking URLs, query parameters, and other messy characteristics of production websites.

WebAsset also includes a JSON-first acquisition underwriting workflow. It estimates normalized cash flow, downside returns, evidence completeness, and a maximum offer. It is decision support, not financial or legal advice, and it does not verify seller claims automatically.

**Philosophy:** Preserve raw observations; curate into analytical representations. Do not equate technical ugliness with low asset value.

## Installation

```shell
python -m venv .venv
source .venv/bin/activate  # or .\.venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
pip install -e .
```

## Usage

### Crawl
```shell
webasset crawl https://example.com --limit 100 -o crawl.json
```

Crawl a website and save raw page data. Captures URLs, titles, headings, text, links, and errors. Handles redirects, off-domain links, and non-HTML content.

### Analyze
```shell
webasset analyze https://example.com --limit 100 -o results.json
```

Complete analysis pipeline: crawl → curate → extract features → semantic analysis → generate candidates → rank recommendations.

### Report
```shell
webasset report results.json
webasset report results.json --filter high --limit 20
```

Print analysis results as human-readable output.

### Underwrite
```shell
webasset underwrite deal.json -o underwriting.json
webasset compare deals.json -o underwriting-comparison.json
```

Deal packets contain `name`, `asset_type`, `asking_price`, and monthly `financials` with `period`, `revenue`, and `operating_expenses`. Optional fields include `transaction_costs`, `working_capital`, `replacement_labor_monthly`, `target_annual_return`, `holding_years`, `evidence`, and `scenarios`.

Evidence values should be `verified`, `reported`, `estimated`, or `missing`. Financials, analytics, bank statements, and transferability must be verified before the model can return `BUY`. The result reports cash-on-cash annual ROI, payback, IRR, NPV at the target return, scenario outcomes, and a downside-based maximum offer; ROI alone does not account for time or risk.

## Pipeline

```
CRAWL
  ↓ Raw observations preserved as-is
RAW DATA  
  ↓ Representation improvement (not correction)
CURATION
  ↓ Language-aware text processing
FEATURE EXTRACTION
  ↓ TF-IDF, clustering, keyword analysis
SEMANTIC ANALYSIS
  ↓ Filter obvious non-matches
CANDIDATE GENERATION
  ↓ Multi-signal scoring
RANKING
  ↓ Output recommendations
REPORT
```

## Architecture

**Data Structures:**
- `RawPage` - Direct crawl output, preserved unchanged
- `CuratedPage` - Analytical representation with metadata
- `Candidate` - Potential recommendation (unscored)
- `ScoredRecommendation` - Ranked with full scoring breakdown
- `AnalysisResult` - Complete analysis with recommendations

**Key Features:**
- Separate raw and curated representations
- Language-aware stopwords (DE, EN, FR, ES)
- Generic page classification (works across site types)
- Hard filters for obvious non-matches only
- Multi-signal ranking (semantic + content quality + type affinity + anchor quality)
- Curation flags as metadata, not exclusion criteria
- Preserves unusual URLs and edge cases

## Configuration

Defaults in code (easily customizable):

```python
# Crawling
limit: 50                    # Max pages to crawl
timeout: 15.0               # Request timeout (seconds)

# Analysis
MIN_SEMANTIC_SIMILARITY: 0.08    # Hard filter for semantic score
MIN_CONTENT_LENGTH: 50          # Hard filter for thin content
MIN_FINAL_SCORE: 0.45           # Hard filter for recommendations (HIGH confidence only)
DUPLICATE_THRESHOLD: 0.85        # Similarity threshold for near-duplicates
```

## Hard Filters

Recommendations are excluded if:

- Source and target are identical
- No shared meaningful vocabulary
- Cannot generate meaningful anchor text
- Target has extremely thin content
- Semantic similarity below 0.08

No arbitrary heuristics - only defensible, observable filters.

## Signals

Recommendations are scored using:

```
semantic_similarity (40%)
+ content_quality (20%)
+ type_affinity (20%)
+ anchor_quality (20%)
```

All signals are normalized (0.0-1.0) and inspectable in output.

## Output Example

```json
{
  "root_url": "https://example.com",
  "crawled_at": "2026-08-28T...",
  "pages": [
    {
      "url": "...",
      "title": "...",
      "classification": "article",
      "keywords": ["term1", "term2", ...],
      "content_quality": 0.85,
      "curation_flags": []
    }
  ],
  "recommendations": [
    {
      "source": "...",
      "target": "...",
      "semantic_similarity": 0.62,
      "final_score": 0.51,
      "anchor_text": "relevant phrase",
      "signals": {
        "semantic": 0.62,
        "content_quality": 0.80,
        "type_affinity": 1.0,
        "anchor_quality": 0.67
      }
    }
  ],
  "statistics": {
    "crawl": {...},
    "curation": {...},
    "semantic": {...}
  }
}
```

## Design Principles

1. **Preserve raw observations** - Don't destructively modify crawl data
2. **Defensible filters** - Only obvious, observable criteria
3. **Language-aware analysis** - Stopwords for multiple languages
4. **Generic classification** - Works across different site types
5. **Inspectable scoring** - All signals visible in output
6. **Fail gracefully** - Handle edge cases without stopping crawl
7. **Validate utility** - "Would a human find this useful?"

## Non-Goals

Not currently:
- A SaaS/web application
- An automatic website editor
- A machine learning model
- A performance analyzer
- A SEO audit tool

WebAsset focuses on the analysis engine as a local CLI tool.

## Development

Run tests/checks:

```shell
python -m compileall webasset
python -m webasset --help
```

Future enhancements (priority order):

1. Better semantic models (sentence embeddings)
2. Contextual anchor placement analysis
3. Click-through/conversion data integration
4. More language support
5. Graph-based ranking (PageRank for link graphs)

Don't optimize heuristics endlessly. Use real crawl results to identify where baseline fails.

"# Web-asset" 
