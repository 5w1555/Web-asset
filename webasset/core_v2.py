"""WebAsset: General-purpose internal link analysis for messy, real-world websites.

Pipeline:
  CRAWL → RAW DATA → CURATION → FEATURE EXTRACTION → SEMANTIC ANALYSIS
  → CANDIDATE GENERATION → RANKING → REPORT

Philosophy:
  - Preserve raw observations; curate into analytical representations.
  - Handle legacy CMS, duplication, boilerplate, tracking URLs, query params.
  - Do not equate technical ugliness with low asset value.
  - Validate: "Would a human actually find this recommendation useful?"
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict, deque
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse, urlunparse
from typing import Optional

import httpx
import trafilatura
from selectolax.parser import HTMLParser
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================================
# LANGUAGE CONFIGURATION
# ============================================================================

STOP_WORDS_BY_LANGUAGE = {
    "de": "aber alle allem allen aller alles als also an auch auf aus bei bin bis das dass dein dem den denn der des die dies diese du durch ein eine einem einen einer eines er es für gegen hat hier ich im in ist ja mit nach nicht noch nur oder sich sie sind über um und von vom vor war was wenn wie wir zu zum zur",
    "en": "a an and are as at be been but by for from had has have he her him his i in into is it its me my no not of on or our she that the their them there they this to was we were what when where which who will with you your",
    "fr": "au aux avec ce ceci cela dans de des du elle en est et eux il ils je la le les leur lui ma mais me mes moi ne nos nous on ou par pas pour que quel quelle quelles quels qui se sont sur un une vos votre vous",
    "es": "a al algo algunos ante con como de del desde el ella ellas ellos en entre es esta este estos hay la las le les lo los más me mi mis no nos o para por que se su sus también un una uno y ya",
}

STOPWORD_SETS = {language: set(words.split()) for language, words in STOP_WORDS_BY_LANGUAGE.items()}
WORD_PATTERN = re.compile(r"[\wäöüß]+", re.IGNORECASE)

# Generic terms that don't meaningfully distinguish pages (language-neutral)
GENERIC_ANCHOR_TERMS = {"page", "site", "visit", "read", "more", "click", "here", "link", "go", "this"}

# ============================================================================
# STAGE 1: DATA STRUCTURES - RAW VS CURATED SEPARATION
# ============================================================================

@dataclass
class RawPage:
    """Direct crawl output - preserved as-is."""
    url: str
    status_code: int = 0
    title: str = ""
    headings: list[str] = field(default_factory=list)
    raw_text: str = ""
    outgoing_links: list[str] = field(default_factory=list)
    error: Optional[str] = None
    redirect_chain: list[str] = field(default_factory=list)
    headers: dict[str, str] = field(default_factory=dict)
    crawled_at: str = ""


@dataclass
class CuratedPage:
    """Analytical representation of a page."""
    url: str
    title: str = ""
    headings: list[str] = field(default_factory=list)
    text: str = ""  # cleaned, normalized text
    language: str = "unknown"
    classification: str = "unknown"
    classification_confidence: float = 0.0
    keywords: list[str] = field(default_factory=list)
    cluster: Optional[int] = None
    incoming_links: list[str] = field(default_factory=list)
    outgoing_links: list[str] = field(default_factory=list)
    
    # Curation metadata
    curation_flags: list[str] = field(default_factory=list)  # "boilerplate", "thin-content", etc.
    is_duplicate_of: Optional[str] = None
    boilerplate_ratio: float = 0.0
    content_quality_score: float = 0.0
    
    # Source tracking
    raw_text_length: int = 0
    error: Optional[str] = None


@dataclass
class Candidate:
    """Potential recommendation before ranking."""
    source: CuratedPage
    target: CuratedPage
    semantic_similarity: float
    shared_keywords: list[str]
    suggested_anchor: str
    is_existing_link: bool = False


@dataclass
class ScoredRecommendation(Candidate):
    """Ranked recommendation with full scoring breakdown."""
    content_quality: float = 0.0
    type_affinity: float = 0.0
    anchor_quality: float = 0.0
    final_score: float = 0.0
    signals: dict[str, float] = field(default_factory=dict)
    explanation: str = ""


@dataclass
class CrawlData:
    """Complete crawl snapshot with raw pages."""
    root_url: str
    pages: list[RawPage]
    crawled_at: str
    crawl_stats: dict = field(default_factory=dict)


@dataclass
class AnalysisResult:
    """Complete analysis with raw, curated, and recommendations."""
    crawl_data: CrawlData
    curated_pages: list[CuratedPage]
    recommendations: list[ScoredRecommendation]
    curation_stats: dict = field(default_factory=dict)
    semantic_stats: dict = field(default_factory=dict)

# ============================================================================
# STAGE 2: URL NORMALIZATION & UTILITIES
# ============================================================================

def normalize_url(url: str) -> str:
    """Normalize URL: defrag, lowercase scheme/host, remove trailing slash."""
    parsed = urlparse(urldefrag(url)[0])
    path = parsed.path or "/"
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    return urlunparse((parsed.scheme.lower(), parsed.netloc.lower(), path, "", parsed.query, ""))


def is_internal(url: str, root_host: str) -> bool:
    """Check if URL is internal to the root domain."""
    host = urlparse(url).netloc.lower().split(":")[0]
    return host == root_host or host.endswith("." + root_host)


def detect_url_pattern(url: str) -> str:
    """Detect special URL patterns."""
    path = urlparse(url).path.lower()
    query = urlparse(url).query.lower()
    
    if "?" in url:
        # Check for common pagination, filtering, tracking patterns
        if any(p in query for p in ["page", "offset", "limit", "paged"]):
            return "pagination"
        if any(p in query for p in ["utm_", "ref=", "tracking", "affiliate"]):
            return "tracking"
        if any(p in query for p in ["filter", "sort", "category_id", "manufacturer"]):
            return "filter"
    
    if "#" in url:
        return "anchor"
    
    return "standard"


def normalize_text(text: str) -> str:
    """Normalize text: Unicode NFKC, deduplicate whitespace, lowercase."""
    text = unicodedata.normalize("NFKC", text).replace("\ufffd", " ")
    return re.sub(r"\s+", " ", text).strip().lower()


def detect_language(text: str) -> str:
    """Detect language based on stopword frequency."""
    words = WORD_PATTERN.findall(text.lower())
    if len(words) < 5:
        return "unknown"
    
    scores = {language: sum(word in stopwords for word in words) for language, stopwords in STOPWORD_SETS.items()}
    language, score = max(scores.items(), key=lambda item: item[1], default=("unknown", 0))
    return language if score >= 2 else "unknown"


def meaningful_terms(text: str, language: str) -> list[str]:
    """Extract terms that are meaningful (not stopwords, not numeric, >2 chars)."""
    stopwords = STOPWORD_SETS.get(language, set()).union(STOPWORD_SETS.get("en", set()))
    return [word for word in WORD_PATTERN.findall(text.lower()) 
            if word not in stopwords and not word.isnumeric() and len(word) > 2]


# ============================================================================
# STAGE 3: CRAWLING
# ============================================================================

def crawl(root_url: str, limit: int = 50, timeout: float = 15.0) -> CrawlData:
    """Crawl website: capture as-is, handle errors gracefully, store raw observations."""
    root_url = normalize_url(root_url)
    parsed_root = urlparse(root_url)
    
    if parsed_root.scheme not in {"http", "https"} or not parsed_root.netloc:
        raise ValueError("URL must include http:// or https:// and a hostname")
    
    pages: dict[str, RawPage] = {}
    queue = deque([root_url])
    visited: set[str] = set()
    headers = {"User-Agent": "webasset/1.0 (+internal link analysis)"}
    
    crawl_stats = {
        "requested": 0,
        "successful": 0,
        "errors": 0,
        "redirects": 0,
        "non_html": 0,
        "off_domain": 0,
    }
    
    with httpx.Client(timeout=timeout, follow_redirects=True, headers=headers) as client:
        while queue and len(pages) < limit:
            url = queue.popleft()
            if url in visited:
                continue
            
            visited.add(url)
            crawl_stats["requested"] += 1
            
            try:
                response = client.get(url)
                final_url = normalize_url(str(response.url))
                
                # Check domain
                if not is_internal(final_url, parsed_root.netloc.lower().split(":")[0]):
                    crawl_stats["off_domain"] += 1
                    continue
                
                # Check content type
                content_type = response.headers.get("content-type", "").lower()
                if "text/html" not in content_type:
                    crawl_stats["non_html"] += 1
                    continue
                
                # Extract page
                page = extract_page(final_url, response.text, response.status_code)
                pages[final_url] = page
                crawl_stats["successful"] += 1
                
                # Queue outgoing links
                for target in page.outgoing_links:
                    if is_internal(target, parsed_root.netloc.lower().split(":")[0]) and target not in visited:
                        queue.append(target)
                
            except httpx.HTTPError as exc:
                pages[url] = RawPage(url=url, error=str(exc))
                crawl_stats["errors"] += 1
    
    # Build incoming links
    page_urls = set(pages)
    for page in pages.values():
        page.outgoing_links = sorted(set(page.outgoing_links) & page_urls)
    
    incoming = defaultdict(list)
    for page in pages.values():
        for target in page.outgoing_links:
            incoming[target].append(page.url)
    
    # Create return data
    from datetime import datetime, timezone
    crawl_data = CrawlData(
        root_url=root_url,
        pages=list(pages.values()),
        crawled_at=datetime.now(timezone.utc).isoformat(),
        crawl_stats=crawl_stats,
    )
    
    return crawl_data


def extract_page(url: str, html: str, status_code: int) -> RawPage:
    """Extract page structure from HTML."""
    tree = HTMLParser(html)
    
    title_elem = tree.css_first("title")
    title = title_elem.text(strip=True) if title_elem else ""
    
    headings = [node.text(strip=True) for node in tree.css("h1, h2, h3") if node.text(strip=True)]
    
    links = []
    for node in tree.css("a[href]"):
        href = node.attributes.get("href", "")
        if href and not href.startswith(("mailto:", "tel:", "javascript:")):
            links.append(normalize_url(urljoin(url, href)))
    
    raw_text = trafilatura.extract(html, include_comments=False, include_tables=False) or ""
    
    return RawPage(
        url=url,
        status_code=status_code,
        title=title,
        headings=headings,
        raw_text=raw_text,
        outgoing_links=sorted(set(links)),
    )


# ============================================================================
# STAGE 4: CURATION - REPRESENTATION IMPROVEMENT, NOT WEBSITE CORRECTION
# ============================================================================

def classify_page(url: str, title: str, headings: list[str], text: str) -> tuple[str, float]:
    """Generic page classification with confidence score. Not definitive - influences analysis."""
    combined = f"{url} {title} {' '.join(headings)} {text[:200]}".lower()
    
    patterns = {
        "homepage": (r"^https?://[^/]+/?$", 0.95),
        "search": (r"(?:search|query|find|results)", 0.85),
        "utility": (r"/(?:contact|about|terms|privacy|sitemap|login|register|account)", 0.9),
        "product": (r"(?:buy|purchase|product|item|sku|model)", 0.75),
        "category": (r"(?:category|collection|browse|listing|catalog)", 0.75),
        "article": (r"(?:blog|post|article|news|guide|tutorial|howto|documentation)", 0.8),
        "service": (r"(?:service|pricing|plan|feature)", 0.7),
    }
    
    for classification, (pattern, confidence) in patterns.items():
        if re.search(pattern, combined):
            return classification, confidence
    
    return "page", 0.5


def content_quality_score(text: str) -> float:
    """Estimate content quality (0.0-1.0) based on length and structure."""
    length = len(text)
    if length < 100:
        return 0.0
    if length < 300:
        return 0.2
    if length < 1000:
        return 0.5
    if length < 3000:
        return 0.8
    return 1.0


def detect_boilerplate(text: str) -> float:
    """Estimate boilerplate ratio by looking for repeated phrases."""
    if not text:
        return 1.0
    
    fragments = [f.strip() for f in re.split(r"(?<=[.!?])\s+", text) if len(f.strip()) >= 20]
    if not fragments:
        return 1.0
    
    fragment_counts = Counter(fragments)
    total_fragments = len(fragments)
    boilerplate_count = sum(count for count in fragment_counts.values() if count >= 2)
    
    return min(1.0, boilerplate_count / max(1, total_fragments))


def curate_raw_page(raw: RawPage) -> CuratedPage:
    """Convert raw page to curated page with analytical metadata."""
    text = normalize_text(raw.raw_text)
    language = detect_language(text)
    classification, confidence = classify_page(raw.url, raw.title, raw.headings, text)
    quality = content_quality_score(text)
    boilerplate = detect_boilerplate(text)
    
    flags = []
    
    # Apply hard filters only when OBVIOUS and DEFENSIBLE
    if raw.error:
        flags.append("error")
    
    # Only flag if clearly a search/utility pattern
    if classification in {"search", "utility"} and confidence >= 0.85:
        flags.append("utility_page")
    
    # Only flag extremely thin content
    if len(text) < 50:
        flags.append("thin_content")
    
    # Very high boilerplate (not moderate boilerplate)
    if boilerplate > 0.8:
        flags.append("high_boilerplate")
    
    # Detect pagination, tracking, etc. - but don't exclude yet
    url_pattern = detect_url_pattern(raw.url)
    if url_pattern in {"tracking"}:
        flags.append(f"url_pattern:{url_pattern}")
    
    return CuratedPage(
        url=raw.url,
        title=raw.title,
        headings=raw.headings,
        text=text,
        language=language,
        classification=classification,
        classification_confidence=confidence,
        curation_flags=flags,
        boilerplate_ratio=boilerplate,
        content_quality_score=quality,
        raw_text_length=len(raw.raw_text),
        error=raw.error,
    )


def detect_duplicates(pages: list[CuratedPage], threshold: float = 0.85) -> dict[str, str]:
    """Detect near-duplicate content. Only mark very similar pages."""
    if len(pages) < 2:
        return {}
    
    duplicate_of = {}
    
    # Use language-aware stopwords
    all_languages = set()
    for page in pages:
        if page.language != "unknown":
            all_languages.add(page.language)
    all_languages.add("en")
    
    custom_stopwords = set()
    for lang in all_languages:
        custom_stopwords.update(STOPWORD_SETS.get(lang, set()))
    
    vectorizer = TfidfVectorizer(stop_words=sorted(custom_stopwords), max_features=1000)
    
    try:
        matrix = vectorizer.fit_transform([p.text for p in pages if p.text])
        similarities = cosine_similarity(matrix)
        
        text_pages = [p for p in pages if p.text]
        for i, page in enumerate(text_pages):
            for j in range(i):
                if similarities[i][j] >= threshold:
                    duplicate_of[page.url] = text_pages[j].url
                    break
    except Exception:
        pass
    
    return duplicate_of


def curate(crawl_data: CrawlData) -> tuple[list[CuratedPage], dict]:
    """Curate raw crawl into analytical representation. Preserve both raw and curated."""
    curated_pages = [curate_raw_page(raw) for raw in crawl_data.pages]
    duplicates = detect_duplicates(curated_pages)
    
    # Mark duplicates
    for url, dup_of in duplicates.items():
        for page in curated_pages:
            if page.url == url:
                page.is_duplicate_of = dup_of
                page.curation_flags.append("near_duplicate")
    
    # Build link lists with curated pages
    url_to_page = {p.url: p for p in curated_pages}
    
    # Set outgoing links from raw data, filtered to existing pages
    for curated_page in curated_pages:
        for raw_page in crawl_data.pages:
            if raw_page.url == curated_page.url:
                curated_page.outgoing_links = [u for u in raw_page.outgoing_links if u in url_to_page]
                break
    
    # Build incoming links
    incoming = defaultdict(list)
    for page in curated_pages:
        for target in page.outgoing_links:
            if target in url_to_page:
                incoming[target].append(page.url)
    
    for page in curated_pages:
        page.incoming_links = incoming[page.url]
    
    stats = {
        "raw_pages": len(crawl_data.pages),
        "curated_pages": len(curated_pages),
        "near_duplicates": len(duplicates),
        "flagged_pages": sum(1 for p in curated_pages if p.curation_flags),
        "usable_for_analysis": len([p for p in curated_pages if not p.curation_flags and p.text]),
    }
    
    return curated_pages, stats


# ============================================================================
# STAGE 5: SEMANTIC ANALYSIS - BASELINE TFIDF
# ============================================================================

def extract_keywords(pages: list[CuratedPage]) -> dict[str, list[str]]:
    """Extract keywords for each page using TF-IDF, language-aware."""
    if not pages or not any(p.text for p in pages):
        return {p.url: [] for p in pages}
    
    # Get language-aware stopwords
    all_languages = set()
    for page in pages:
        if page.language != "unknown":
            all_languages.add(page.language)
    all_languages.add("en")  # Always include English
    
    custom_stopwords = set()
    for lang in all_languages:
        custom_stopwords.update(STOPWORD_SETS.get(lang, set()))
    
    vectorizer = TfidfVectorizer(
        stop_words=sorted(custom_stopwords),
        ngram_range=(1, 2),
        max_features=3000,
        min_df=1,
        max_df=0.8
    )
    
    texts = [p.text for p in pages]
    
    try:
        matrix = vectorizer.fit_transform(texts)
        terms = vectorizer.get_feature_names_out()
        
        keywords = {}
        for idx, page in enumerate(pages):
            row = matrix[idx].toarray()[0]
            top_indices = row.argsort()[::-1][:10]
            keywords[page.url] = [terms[i] for i in top_indices if row[i] > 0]
        return keywords
    except Exception:
        return {p.url: [] for p in pages}


def semantic_analysis(curated_pages: list[CuratedPage]) -> dict:
    """Perform semantic analysis: TF-IDF, clustering, keyword extraction."""
    usable = [p for p in curated_pages if p.text and not p.curation_flags]
    
    if not usable:
        return {"pages": len(usable), "keywords": {}, "clusters": 0, "similarity_matrix": None}
    
    keywords = extract_keywords(usable)
    for page in usable:
        page.keywords = keywords.get(page.url, [])
    
    # Clustering
    if len(usable) >= 2:
        try:
            # Language-aware stopwords for vectorizer
            all_languages = set()
            for page in usable:
                if page.language != "unknown":
                    all_languages.add(page.language)
            all_languages.add("en")
            
            custom_stopwords = set()
            for lang in all_languages:
                custom_stopwords.update(STOPWORD_SETS.get(lang, set()))
            
            vectorizer = TfidfVectorizer(
                stop_words=sorted(custom_stopwords),
                max_features=5000,
                min_df=1,
                max_df=0.8
            )
            matrix = vectorizer.fit_transform([p.text for p in usable])
            
            cluster_count = min(max(2, round(len(usable) ** 0.5)), len(usable))
            labels = KMeans(n_clusters=cluster_count, n_init=10, random_state=42).fit_predict(matrix)
            
            for page, label in zip(usable, labels):
                page.cluster = int(label)
        except Exception:
            pass
    
    return {
        "pages": len(usable),
        "keywords_extracted": sum(1 for p in usable if p.keywords),
        "clusters": len(set(p.cluster for p in usable if p.cluster is not None)),
    }


# ============================================================================
# STAGE 6: CANDIDATE GENERATION - SEPARATE FROM RANKING
# ============================================================================

def generate_candidates(usable_pages: list[CuratedPage]) -> list[Candidate]:
    """Generate potential recommendations. No scoring yet - just filtering obvious non-matches."""
    if len(usable_pages) < 2:
        return []
    
    candidates = []
    
    # Compute similarity once with language-aware stopwords
    all_languages = set()
    for page in usable_pages:
        if page.language != "unknown":
            all_languages.add(page.language)
    all_languages.add("en")
    
    custom_stopwords = set()
    for lang in all_languages:
        custom_stopwords.update(STOPWORD_SETS.get(lang, set()))
    
    vectorizer = TfidfVectorizer(
        stop_words=sorted(custom_stopwords),
        max_features=5000,
        min_df=1,
        max_df=0.8
    )
    matrix = vectorizer.fit_transform([p.text for p in usable_pages])
    similarities = cosine_similarity(matrix)
    
    for src_idx, source in enumerate(usable_pages):
        existing = set(source.outgoing_links)
        
        for tgt_idx, target in enumerate(usable_pages):
            if src_idx == tgt_idx or target.url in existing:
                continue
            
            similarity = similarities[src_idx][tgt_idx]
            
            # Hard filter 1: Minimum semantic similarity (obviously not related)
            if similarity < 0.08:
                continue
            
            # Hard filter 2: Must have shared vocabulary
            shared = sorted(set(source.keywords) & set(target.keywords))
            if not shared:
                continue
            
            # Hard filter 3: Must be able to generate meaningful anchor
            anchor = generate_anchor(target, shared)
            if not anchor:
                continue
            
            candidates.append(Candidate(
                source=source,
                target=target,
                semantic_similarity=similarity,
                shared_keywords=shared,
                suggested_anchor=anchor,
                is_existing_link=target.url in existing,
            ))
    
    return candidates


def generate_anchor(target: CuratedPage, shared_keywords: list[str]) -> str:
    """Generate meaningful anchor text from target's subject matter."""
    language = target.language
    
    # Prefer terms from title if meaningful
    title_terms = [t for t in meaningful_terms(target.title, language) 
                   if t not in GENERIC_ANCHOR_TERMS and t in target.keywords]
    
    # Fallback to shared keywords
    candidates = title_terms or [t for t in shared_keywords if t in target.keywords and t not in GENERIC_ANCHOR_TERMS]
    
    if len(candidates) < 2:
        return ""
    
    anchor = " ".join(candidates[:4]).strip()
    
    if len(anchor) < 5:  # Too short
        return ""
    
    return anchor


# ============================================================================
# STAGE 7: RANKING - MULTIPLE SIGNALS
# ============================================================================

def score_recommendations(candidates: list[Candidate]) -> list[ScoredRecommendation]:
    """Rank candidates using multiple signals. Keep scoring model inspectable."""
    scored = []
    
    for candidate in candidates:
        source, target = candidate.source, candidate.target
        
        # Signal 1: Content quality of target
        content_quality = target.content_quality_score
        
        # Signal 2: Type affinity (same-type pages should link)
        type_affinity = 1.0 if source.classification == target.classification else 0.75
        
        # Signal 3: Anchor quality (meaningful terms in suggested anchor)
        anchor_terms = meaningful_terms(candidate.suggested_anchor, target.language)
        anchor_quality = min(1.0, len(anchor_terms) / 3.0)
        
        # Signal 4: Semantic similarity (baseline)
        semantic_score = candidate.semantic_similarity
        
        # Combine signals
        final_score = (
            semantic_score * 0.4 +
            content_quality * 0.2 +
            type_affinity * 0.2 +
            anchor_quality * 0.2
        )
        
        # Determine confidence
        confidence = "HIGH" if final_score >= 0.45 else "MEDIUM" if final_score >= 0.25 else "LOW"
        
        explanation = f"Semantic similarity: {semantic_score:.1%} | Shared terms: {', '.join(candidate.shared_keywords[:3])}"
        
        rec = ScoredRecommendation(
            source=candidate.source,
            target=candidate.target,
            semantic_similarity=candidate.semantic_similarity,
            shared_keywords=candidate.shared_keywords,
            suggested_anchor=candidate.suggested_anchor,
            is_existing_link=candidate.is_existing_link,
            content_quality=content_quality,
            type_affinity=type_affinity,
            anchor_quality=anchor_quality,
            final_score=final_score,
            signals={
                "semantic": semantic_score,
                "content_quality": content_quality,
                "type_affinity": type_affinity,
                "anchor_quality": anchor_quality,
            },
            explanation=explanation,
        )
        scored.append(rec)
    
    return sorted(scored, key=lambda r: r.final_score, reverse=True)


# ============================================================================
# STAGE 8: FULL ANALYSIS PIPELINE
# ============================================================================

def analyze(crawl_data: CrawlData) -> AnalysisResult:
    """Run full analysis pipeline."""
    # Curation
    curated_pages, curation_stats = curate(crawl_data)
    
    # Semantic analysis - note: we use pages with content, even if they're marked duplicate
    # The flags are metadata, not exclusion criteria
    usable = [p for p in curated_pages if p.text and "error" not in p.curation_flags]
    
    # Semantic analysis
    semantic_stats = semantic_analysis(usable)  # Pass all usable, not just unflagged
    
    # Generate candidates from unflagged pages only (hard errors only)
    unflagged = [p for p in usable if not p.curation_flags]
    candidates = generate_candidates(unflagged)
    
    # Score recommendations - only HIGH confidence
    all_scored = score_recommendations(candidates)
    high_confidence = [r for r in all_scored if r.final_score >= 0.45]
    
    # Limit to top 2 per source
    recommendations = []
    by_source = defaultdict(list)
    for rec in high_confidence:
        by_source[rec.source.url].append(rec)
    
    for recs in by_source.values():
        recommendations.extend(recs[:2])
    
    recommendations.sort(key=lambda r: r.final_score, reverse=True)
    
    return AnalysisResult(
        crawl_data=crawl_data,
        curated_pages=curated_pages,
        recommendations=recommendations,
        curation_stats=curation_stats,
        semantic_stats=semantic_stats,
    )


# ============================================================================
# SERIALIZATION
# ============================================================================

def save_json(obj: object, path: str | Path) -> None:
    """Save object as JSON."""
    def serializer(o):
        if hasattr(o, '__dataclass_fields__'):
            return asdict(o)
        raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")
    
    Path(path).write_text(json.dumps(obj, default=serializer, indent=2, ensure_ascii=False), encoding="utf-8")


def analysis_to_dict(result: AnalysisResult) -> dict:
    """Convert AnalysisResult to JSON-serializable dict."""
    return {
        "root_url": result.crawl_data.root_url,
        "crawled_at": result.crawl_data.crawled_at,
        "pages": [
            {
                "url": p.url,
                "title": p.title,
                "headings": p.headings,
                "keywords": p.keywords,
                "cluster": p.cluster,
                "classification": p.classification,
                "language": p.language,
                "incoming_links": len(p.incoming_links),
                "outgoing_links": len(p.outgoing_links),
                "content_length": len(p.text),
                "content_quality": round(p.content_quality_score, 3),
                "boilerplate_ratio": round(p.boilerplate_ratio, 3),
                "curation_flags": p.curation_flags,
            }
            for p in result.curated_pages
            if not p.curation_flags
        ],
        "recommendations": [
            {
                "source": r.source.url,
                "target": r.target.url,
                "semantic_similarity": round(r.semantic_similarity, 4),
                "similarity_percent": round(r.semantic_similarity * 100, 1),
                "final_score": round(r.final_score, 4),
                "score_percent": round(r.final_score * 100, 1),
                "anchor_text": r.suggested_anchor,
                "shared_keywords": r.shared_keywords[:5],
                "signals": {k: round(v, 3) for k, v in r.signals.items()},
                "explanation": r.explanation,
            }
            for r in result.recommendations
        ],
        "statistics": {
            "crawl": result.crawl_data.crawl_stats,
            "curation": result.curation_stats,
            "semantic": result.semantic_stats,
            "recommendations": {
                "total": len(result.recommendations),
                "high_confidence": sum(1 for r in result.recommendations if r.final_score >= 0.45),
            }
        }
    }
