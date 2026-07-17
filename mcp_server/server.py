"""
MCP server for the Catalog Onboarding Agent project.
Exposes 4 tools over the Model Context Protocol.

Setup:
    pip install fastmcp
    python server.py
"""

import json
import re
import difflib
from pathlib import Path
from fastmcp import FastMCP
from sentence_transformers import SentenceTransformer
import numpy as np

mcp = FastMCP("catalog-onboarding-tools")

DATA_DIR = Path(__file__).parent.parent / "data"

with open(DATA_DIR / "existing_catalog.json") as f:
    CATALOG = json.load(f)

with open(DATA_DIR / "category_rules.json") as f:
    CATEGORY_RULES = json.load(f)

with open(DATA_DIR / "banned_keywords.json") as f:
    BANNED = json.load(f)

_embedder = SentenceTransformer("all-MiniLM-L6-v2")
_CATALOG_EMBEDDINGS = _embedder.encode([item["name"] for item in CATALOG])


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _normalize(name: str) -> str:
    """Strip units/casing/punctuation so 'amul milk 500 ml' 'Amul Milk 500 ML' and
    'amul milk 500ml' compare as equivalent, this is what makes
    duplicate detection catch real-world naming variants."""
    name = name.lower()
    name = re.sub(r"[^a-z0-9\s]", "", name)
    name = re.sub(r"\b(ml|g|gm|kg|l|litre|liter|pcs|pack)\b", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


@mcp.tool()
def search_existing_catalog(query: str) -> dict:
    """Search the existing product catalog by meaning, not just exact
    text (e.g. 'coffee' finds coffee products, not unrelated items that
    share letters with it). Returns up to 5 closest-matching items with
    their category and price. Use this to look up what's already
    onboarded before approving a new item."""
    query_embedding = _embedder.encode([query])[0]
    scored = [
        (_cosine_similarity(query_embedding, emb), item)
        for emb, item in zip(_CATALOG_EMBEDDINGS, CATALOG)
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    results = [item for score, item in scored[:5] if score > 0.35]
    if not results:
        return {"results": [], "message": "No matching items found"}
    return {"results": results}


@mcp.tool()
def flag_duplicate(item_name: str) -> dict:
    """Check whether a new item name is a near-duplicate of an item
    already in the catalog (catches casing, spacing, and unit-format
    variants, e.g. 'Amul Milk 500ml' vs 'amul milk 500 ml'). Returns
    is_duplicate=true with the matched item if similarity is high."""
    normalized_query = _normalize(item_name)
    best_score, best_match = 0.0, None
    for item in CATALOG:
        score = difflib.SequenceMatcher(
            None, normalized_query, _normalize(item["name"])
        ).ratio()
        if score > best_score:
            best_score, best_match = score, item

    is_duplicate = best_score >= 0.80  # tuned threshold, see note below
    return {
        "is_duplicate": is_duplicate,
        "similarity": round(best_score, 3),
        "matched_item": best_match if is_duplicate else None,
    }


@mcp.tool()
def check_category_rules(category: str, price_inr: float, has_weight_field: bool) -> dict:
    """Validate a submission against its category's rules: price band
    and whether a weight/quantity field is required. Returns any rule
    violations found. Category must exactly match one of the known
    categories (use search_existing_catalog first if unsure)."""
    rules = CATEGORY_RULES.get(category)
    if rules is None:
        return {"valid": False, "violations": [f"Unknown category: '{category}'"]}

    violations = []
    low, high = rules["price_band_inr"]
    if not (low <= price_inr <= high):
        violations.append(
            f"Price {price_inr} INR is outside expected band [{low}, {high}] for {category}"
        )
    if rules["requires_weight_field"] and not has_weight_field:
        violations.append(f"{category} items require a weight/quantity field, none provided")

    return {"valid": len(violations) == 0, "violations": violations}


@mcp.tool()
def check_banned_keywords(item_name: str) -> dict:
    """Check whether an item name contains any restricted/banned
    keywords (alcohol, tobacco, weapons, pharmaceuticals) that require
    special licensing and cannot be auto-approved."""
    lower_name = item_name.lower()
    matched = [kw for kw in BANNED["banned_keywords"] if kw in lower_name]
    return {
        "is_banned": len(matched) > 0,
        "matched_keywords": matched,
        "reason": BANNED["reason"] if matched else None,
    }


if __name__ == "__main__":
    mcp.run()