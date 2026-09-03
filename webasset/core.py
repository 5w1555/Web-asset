from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict, deque
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse, urlunparse

import httpx
import trafilatura
from selectolax.parser import HTMLParser
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

STOP_WORDS_BY_LANGUAGE = {
    "de": "aber alle allem allen aller alles als also an auch auf aus bei bin bis das dass dein dem den denn der des die dies diese du durch ein eine einem einen einer eines er es für gegen hat hier ich im in ist ja mit nach nicht noch nur oder sich sie sind über um und von vom vor war was wenn wie wir zu zum zur",
    "en": "a an and are as at be been but by for from had has have he her him his i in into is it its me my no not of on or our she that the their them there they this to was we were what when where which who will with you your",
    "fr": "au aux avec ce ceci cela dans de des du elle en est et eux il ils je la le les leur lui ma mais me mes moi ne nos nous on ou par pas pour que quel quelle quelles quels qui se sont sur un une vos votre vous",
    "es": "a al algo algunos ante con como de del desde el ella ellas ellos en entre es esta este estos hay la las le les lo los más me mi mis no nos o para por que se su sus también un una uno y ya",
}
STOPWORD_SETS = {language: set(words.split()) for language, words in STOP_WORDS_BY_LANGUAGE.items()}
WORD_PATTERN = re.compile(r"[\wäöüß]+", re.IGNORECASE)
GENERIC_ANCHOR_TERMS = {"sportimed", "kaufen", "online", "jetzt", "bestellen", "shop", "de"}


@dataclass
class Page:
    url: str
    status_code: int = 0
    title: str = ""
    headings: list[str] = field(default_factory=list)
    text: str = ""
    outgoing_links: list[str] = field(default_factory=list)
    incoming_links: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    cluster: int | None = None
    error: str | None = None
    language: str = "unknown"
    classification: str = "page"
    curation_reasons: list[str] = field(default_factory=list)


@dataclass
class CrawlData:
    root_url: str
    pages: list[Page]
    crawled_at: str


@dataclass
class CurationResult:
    data: CrawlData
    removed_boilerplate: dict[str, list[str]]
    duplicate_of: dict[str, str]


def normalize_url(url: str) -> str:
    parsed = urlparse(urldefrag(url)[0])
    path = parsed.path or "/"
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    return urlunparse((parsed.scheme.lower(), parsed.netloc.lower(), path, "", parsed.query, ""))


def is_internal(url: str, root_host: str) -> bool:
    host = urlparse(url).netloc.lower().split(":")[0]
    return host == root_host or host.endswith("." + root_host)


def extract_page(url: str, html: str, status_code: int) -> Page:
    tree = HTMLParser(html)
    title = tree.css_first("title")
    headings = [node.text(strip=True) for node in tree.css("h1, h2, h3") if node.text(strip=True)]
    links = []
    for node in tree.css("a[href]"):
        href = node.attributes.get("href", "")
        if href and not href.startswith(("mailto:", "tel:", "javascript:")):
            links.append((normalize_url(urljoin(url, href)), node.text(strip=True)))

    text = trafilatura.extract(html, include_comments=False, include_tables=False) or ""
    return Page(
        url=url,
        status_code=status_code,
        title=title.text(strip=True) if title else "",
        headings=headings,
        text=normalize_text(text),
        outgoing_links=sorted({target for target, _ in links}),
    )


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("\ufffd", " ")
    return re.sub(r"\s+", " ", text).strip().lower()


def detect_language(text: str) -> str:
    words = WORD_PATTERN.findall(text.lower())
    scores = {language: sum(word in stopwords for word in words) for language, stopwords in STOPWORD_SETS.items()}
    language, score = max(scores.items(), key=lambda item: item[1], default=("unknown", 0))
    return language if score >= 2 else "unknown"


def meaningful_terms(text: str, language: str) -> list[str]:
    stopwords = STOPWORD_SETS.get(language, set()).union(STOPWORD_SETS["en"])
    return [word for word in WORD_PATTERN.findall(text.lower()) if word not in stopwords and not word.isnumeric() and len(word) > 2]


def vectorizer_for(pages: list[Page], **kwargs) -> TfidfVectorizer:
    stopwords = set().union(*(STOPWORD_SETS.get(page.language, set()) for page in pages), STOPWORD_SETS["en"])
    if pages:
        document_frequency = Counter(
            term for page in pages for term in set(meaningful_terms(page.text, page.language))
        )
        common_limit = max(2, round(len(pages) * 0.6))
        stopwords.update(term for term, count in document_frequency.items() if count >= common_limit)
    return TfidfVectorizer(stop_words=sorted(stopwords), **kwargs)


UTILITY_PATTERNS = re.compile(
    r"/(?:advanced_search|search|login|log-in|register|account|cart|checkout|wishlist|compare|sitemap|contact)(?:[./?]|$)",
    re.IGNORECASE,
)
BOILERPLATE_PHRASES = (
    "newsletter abonnieren",
    "hilfe & support",
    "zahlungsweisen",
    "versanddienstleister",
    "webshop erstellen",
    "ihre meinung",
    "frage zum produkt",
)


def _content_fragments(text: str) -> list[str]:
    return [fragment.strip() for fragment in re.split(r"(?<=[.!?])\s+|\s+-\s+", text) if len(fragment.strip()) >= 35]


def classify_page(page: Page) -> str:
    value = f"{page.url} {page.title} {' '.join(page.headings)}".lower()
    
    # Reject utility/search/system pages
    if UTILITY_PATTERNS.search(urlparse(page.url).path) or any(word in value for word in ("suchergebnisse", "erweiterte suche", "kontakt", "seitenübersicht")):
        return "search" if "search" in value or "suche" in value else "utility"
    
    # Reject filter/category pages with query parameters or manufacturers IDs
    if "?" in page.url or "manufacturers_id" in page.url:
        return "category"
    
    # Product pages (specific)
    if ".html" in page.url and any(word in value for word in ("kaufen", "shop", "produkt", "set", "modell")):
        return "product"
    
    # Category/collection pages (should be rare in final output)
    if any(word in value for word in ("kategorie", "sortiment", "online kaufen")):
        return "category"
    
    # Content/article pages
    if any(word in value for word in ("blog", "ratgeber", "anleitung", "test", "newcomer", "guide", "howto", "tutorial")):
        return "article"
    
    return "page"


def curate(data: CrawlData, duplicate_threshold: float = 0.92) -> CurationResult:
    """Create an analysis copy while leaving the raw CrawlData untouched."""
    curated = deepcopy(data)
    pages = [page for page in curated.pages if not page.error]
    fragment_counts = Counter(fragment for page in pages for fragment in _content_fragments(page.text))
    page_count = max(1, len(pages))
    repeated = {fragment for fragment, count in fragment_counts.items() if count >= 3 or count / page_count >= 0.35}
    removed: dict[str, list[str]] = {}
    duplicate_of: dict[str, str] = {}

    for page in curated.pages:
        page.text = normalize_text(page.text)
        page.language = detect_language(page.text)
        original_fragments = _content_fragments(page.text)
        kept_fragments = [fragment for fragment in original_fragments if fragment not in repeated and not any(phrase in fragment for phrase in BOILERPLATE_PHRASES)]
        if original_fragments and len(kept_fragments) != len(original_fragments):
            removed[page.url] = [fragment for fragment in original_fragments if fragment not in kept_fragments]
        page.text = " ".join(kept_fragments) or page.text
        page.classification = classify_page(page)
        reasons = []
        if page.classification in {"utility", "search", "category"}:
            reasons.append("utility/search/category page")
        # Stricter threshold: need at least 300 chars for meaningful content
        if len(page.text) < 300:
            reasons.append("insufficient meaningful content")
        page.curation_reasons = reasons

    content_pages = [page for page in curated.pages if page.text and not page.error]
    if len(content_pages) > 1:
        matrix = vectorizer_for(content_pages, max_features=5000).fit_transform([page.text for page in content_pages])
        similarities = cosine_similarity(matrix)
        for index, page in enumerate(content_pages):
            for other_index in range(index):
                if similarities[index][other_index] >= duplicate_threshold:
                    duplicate_of[page.url] = content_pages[other_index].url
                    page.curation_reasons.append("near-duplicate content")
                    break
    return CurationResult(curated, removed, duplicate_of)


def crawl(root_url: str, limit: int = 50, timeout: float = 15.0) -> CrawlData:
    root_url = normalize_url(root_url)
    parsed_root = urlparse(root_url)
    if parsed_root.scheme not in {"http", "https"} or not parsed_root.netloc:
        raise ValueError("URL must include http:// or https:// and a hostname")

    pages: dict[str, Page] = {}
    queue = deque([root_url])
    visited: set[str] = set()
    headers = {"User-Agent": "webasset/0.1 (+local internal-link analysis)"}
    with httpx.Client(timeout=timeout, follow_redirects=True, headers=headers) as client:
        while queue and len(pages) < limit:
            url = queue.popleft()
            if url in visited:
                continue
            visited.add(url)
            try:
                response = client.get(url)
                final_url = normalize_url(str(response.url))
                if not is_internal(final_url, parsed_root.netloc.lower().split(":")[0]):
                    continue
                content_type = response.headers.get("content-type", "")
                if "text/html" not in content_type:
                    continue
                page = extract_page(final_url, response.text, response.status_code)
                pages[final_url] = page
                for target in page.outgoing_links:
                    if is_internal(target, parsed_root.netloc.lower().split(":")[0]) and target not in visited:
                        queue.append(target)
            except httpx.HTTPError as exc:
                pages[url] = Page(url=url, error=str(exc))

    page_urls = set(pages)
    for page in pages.values():
        page.outgoing_links = sorted(set(page.outgoing_links) & page_urls)
    incoming = defaultdict(list)
    for page in pages.values():
        for target in page.outgoing_links:
            incoming[target].append(page.url)
    for page in pages.values():
        page.incoming_links = sorted(incoming[page.url])
    from datetime import datetime, timezone
    return CrawlData(root_url=root_url, pages=list(pages.values()), crawled_at=datetime.now(timezone.utc).isoformat())


def analyze(data: CrawlData, use_embeddings: bool = False) -> dict:
    curation = curate(data)
    curated_data = curation.data
    usable = [
        page for page in curated_data.pages
        if page.text and not page.error and not page.curation_reasons
    ]
    corpus = [page.text for page in usable]
    similarity = []
    vocabulary: list[str] = []
    matrix = None
    if corpus:
        tfidf = vectorizer_for(usable, max_features=5000)
        matrix = tfidf.fit_transform(corpus)
        vocabulary = sorted(tfidf.get_feature_names_out().tolist())
        similarity = cosine_similarity(matrix)
    terms_by_page = extract_keywords(usable)
    for page, terms in zip(usable, terms_by_page):
        page.keywords = terms

    if len(usable) >= 2:
        cluster_count = min(max(2, round(len(usable) ** 0.5)), len(usable))
        labels = KMeans(n_clusters=cluster_count, n_init=10, random_state=7).fit_predict(matrix)
        for page, label in zip(usable, labels):
            page.cluster = int(label)

    recommendations = []
    for source_index, source in enumerate(usable):
        existing = set(source.outgoing_links)
        candidates = []
        for target_index, target in enumerate(usable):
            if source.url == target.url or target.url in existing:
                continue
            semantic_score = float(similarity[source_index][target_index])
            overlap = sorted(set(source.keywords) & set(target.keywords))
            anchor = make_anchor(target, overlap)
            
            # HARD FILTER 1: Semantic similarity threshold
            if semantic_score < 0.15:
                continue
            
            # HARD FILTER 2: Meaningful vocabulary (not just stopwords)
            if not overlap or len(overlap) < 2:
                continue
            
            # HARD FILTER 3: Anchor must have meaningful terms
            anchor_terms = meaningful_terms(anchor, target.language)
            if not anchor or len(anchor_terms) < 2:
                continue
            
            # HARD FILTER 4: Target must have substantial content
            if len(target.text) < 300:
                continue
            
            content_quality = min(1.0, len(target.text) / 1200)
            page_relevance = 1.0 if source.classification == target.classification else 0.75
            anchor_quality = min(1.0, len(anchor_terms) / 3)
            score = semantic_score * content_quality * page_relevance * anchor_quality
            
            # HARD FILTER 5: Recommendation score must be high
            if score < 0.30:
                continue
            
            candidates.append((score, semantic_score, target, overlap, anchor))
        
        # PRECISION: Only top 2 candidates per source, all must be HIGH confidence
        for score, semantic_score, target, overlap, anchor in sorted(candidates, key=lambda item: item[0], reverse=True)[:2]:
            if score < 0.45:
                continue
            recommendations.append({
                "source": source.url,
                "target": target.url,
                "similarity": round(semantic_score, 4),
                "similarity_percent": round(semantic_score * 100, 1),
                "recommendation_score": round(score, 4),
                "recommendation_percent": round(score * 100, 1),
                "direction": f"{source.url} -> {target.url}",
                "anchor_text": anchor,
                "confidence": "HIGH",
                "reason": "Subject-specific vocabulary match: " + ", ".join(overlap[:3]),
            })

    # Output only curated, non-flagged pages
    page_rows = []
    for page in curated_data.pages:
        if page.error or page.curation_reasons:
            continue
        importance = page_rank(page, curated_data.pages)
        page_rows.append({
            "url": page.url, "title": page.title, "headings": page.headings,
            "keywords": page.keywords, "cluster": page.cluster,
            "incoming_links": len(page.incoming_links), "outgoing_links": len(page.outgoing_links),
            "importance": round(importance, 4), "orphan": len(page.incoming_links) == 0,
            "classification": page.classification,
            "language": page.language,
            "curated_text_length": len(page.text),
        })
    
    return {
        "root_url": data.root_url,
        "crawled_at": data.crawled_at,
        "pages": page_rows,
        "recommendations": recommendations,
        "tfidf_vocabulary": vocabulary,
        "curation": {
            "raw_pages": len(data.pages),
            "analyzed_pages": len(usable),
            "pages_in_output": len(page_rows),
            "boilerplate_fragments_removed": sum(len(fragments) for fragments in curation.removed_boilerplate.values()),
            "near_duplicates": len(curation.duplicate_of),
            "raw_crawl_preserved_separately": True,
        },
        "embedding_enabled": use_embeddings,
        "embedding_note": "Embedding provider hook reserved; TF-IDF is used in this MVP.",
    }


def extract_keywords(pages: list[Page]) -> list[list[str]]:
    if not pages:
        return []
    vectorizer = vectorizer_for(pages, ngram_range=(1, 2), max_features=3000)
    matrix = vectorizer.fit_transform([page.text for page in pages])
    terms = vectorizer.get_feature_names_out()
    return [[terms[index] for index in row.argsort()[::-1][:8] if matrix[row_index, index] > 0] for row_index, row in enumerate(matrix.toarray())]


def make_anchor(target: Page, overlap: list[str]) -> str:
    """Generate anchor text from meaningful target concepts.
    
    Rejects:
    - stopword-only anchors
    - numeric anchors
    - generic fragments
    - full page titles when inappropriate
    
    Prefers concise, natural phrases derived from target's actual subject.
    """
    title_terms = [term for term in meaningful_terms(target.title, target.language) if term not in GENERIC_ANCHOR_TERMS]
    target_terms = set(meaningful_terms(target.text, target.language))
    title_terms = [term for term in title_terms if term in target_terms]
    
    # Prefer meaningful title terms, fallback to overlap terms
    candidates = title_terms or [term for term in overlap if term in target_terms and term not in GENERIC_ANCHOR_TERMS]
    
    # Require at least 2 meaningful terms for a valid anchor
    if len(candidates) < 2:
        return ""
    
    # Build anchor: take top meaningful terms
    anchor = " ".join(candidates[:4]).strip()
    
    # Reject if too short or all numeric
    if not anchor or len(anchor) < 6 or not any(not token.isnumeric() for token in candidates):
        return ""
    
    return anchor


def page_rank(page: Page, pages: list[Page]) -> float:
    total = len(pages) or 1
    incoming = len(page.incoming_links) / total
    outgoing = len(page.outgoing_links) / total
    return min(1.0, 0.65 * incoming + 0.35 * outgoing)


def save_json(value: object, path: str | Path) -> None:
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=True), encoding="utf-8")


def load_crawl(path: str | Path) -> CrawlData:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return CrawlData(root_url=raw["root_url"], crawled_at=raw["crawled_at"], pages=[Page(**page) for page in raw["pages"]])


def crawl_to_dict(data: CrawlData) -> dict:
    return {"root_url": data.root_url, "crawled_at": data.crawled_at, "pages": [asdict(page) for page in data.pages]}


def curation_to_dict(result: CurationResult) -> dict:
    pages = []
    for page in result.data.pages:
        row = asdict(page)
        row.update({
            "classification": page.classification,
            "curation_reasons": page.curation_reasons,
            "duplicate_of": result.duplicate_of.get(page.url),
        })
        pages.append(row)
    return {
        "root_url": result.data.root_url,
        "crawled_at": result.data.crawled_at,
        "curated_from": "raw crawl snapshot",
        "pages": pages,
        "removed_boilerplate": result.removed_boilerplate,
        "near_duplicates": result.duplicate_of,
    }
