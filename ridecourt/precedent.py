"""Precedent retrieval — simple TF-IDF keyword similarity.

Retrieves the 3 most similar past rulings from ``precedents.json``
using category matching and keyword TF-IDF cosine similarity over the
precedent summary and the dispute's claim text.

No external vector DB — pure Python, in-memory.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Sequence

from .data import Dispute, Precedent


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "is", "was", "were", "are",
    "to", "of", "in", "on", "at", "by", "for", "with", "from", "that",
    "this", "it", "he", "she", "they", "we", "you", "i", "no", "not",
    "yes", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "can",
    "the", "driver", "rider", "trip", "ride", "for", "s$", "sgd",
})

_WORD_RE = re.compile(r"[a-z]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alpha, remove stop words."""
    tokens = _WORD_RE.findall(text.lower())
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]


# ---------------------------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------------------------


@dataclass
class PrecedentMatch:
    """A retrieved precedent with similarity score and reason."""

    case_id: str
    category: str
    summary: str
    outcome: str
    amount: float
    similarity: float
    reason: str
    source: str = "seed"  # "seed" or "override"


def _term_freq(tokens: list[str]) -> dict[str, float]:
    """Compute term frequency for a token list."""
    if not tokens:
        return {}
    counts: dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    total = len(tokens)
    return {t: c / total for t, c in counts.items()}


def _idf(doc_tokens: list[list[str]]) -> dict[str, float]:
    """Compute inverse document frequency."""
    n = len(doc_tokens)
    if n == 0:
        return {}
    df: dict[str, int] = {}
    for tokens in doc_tokens:
        for t in set(tokens):
            df[t] = df.get(t, 0) + 1
    return {
        t: math.log((1 + n) / (1 + d)) + 1.0
        for t, d in df.items()
    }


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosine similarity between two sparse vectors."""
    if not a or not b:
        return 0.0
    dot = sum(a[t] * b.get(t, 0.0) for t in a if t in b)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# Category match weight (0 or 1)
_CATEGORY_WEIGHT = 0.3
# Text similarity weight
_TEXT_WEIGHT = 0.7


def retrieve_precedents(
    dispute: Dispute,
    precedents: Sequence[Precedent],
    top_k: int = 3,
) -> list[PrecedentMatch]:
    """Retrieve the top-k most similar precedents for a dispute.

    Similarity = category_match * 0.3 + tfidf_cosine * 0.7

    Category match: same category → 1.0, else 0.0.
    TF-IDF cosine: over (precedent.summary + dispute.claim_text).
    """
    if not precedents:
        return []

    # Build document tokens for IDF: one "document" per precedent
    # summary, plus the dispute claim text.
    prec_tokens_list = [_tokenize(p.summary) for p in precedents]
    dispute_tokens = _tokenize(dispute.claim_text)
    all_docs = prec_tokens_list + [dispute_tokens]
    idf = _idf(all_docs)

    # TF-IDF vectors for each precedent and the dispute.
    prec_tfidf: list[dict[str, float]] = []
    for tokens in prec_tokens_list:
        tf = _term_freq(tokens)
        prec_tfidf.append({t: tf_val * idf.get(t, 0.0) for t, tf_val in tf.items()})

    dispute_tf = _term_freq(dispute_tokens)
    dispute_tfidf = {
        t: tf_val * idf.get(t, 0.0) for t, tf_val in dispute_tf.items()
    }

    # Compute similarity for each precedent.
    matches: list[PrecedentMatch] = []
    for i, p in enumerate(precedents):
        cat_match = 1.0 if p.category == dispute.category else 0.0
        text_sim = _cosine(prec_tfidf[i], dispute_tfidf)
        similarity = _CATEGORY_WEIGHT * cat_match + _TEXT_WEIGHT * text_sim

        # Build a human-readable reason.
        reason_parts: list[str] = []
        if cat_match:
            reason_parts.append("same category")
        if text_sim > 0.01:
            reason_parts.append(f"keyword overlap {text_sim:.2f}")
        if not reason_parts:
            reason_parts.append("no strong match")
        reason = "; ".join(reason_parts)

        source = getattr(p, "source", "seed") or "seed"
        matches.append(PrecedentMatch(
            case_id=p.case_id,
            category=p.category,
            summary=p.summary,
            outcome=p.outcome,
            amount=p.amount,
            similarity=round(similarity, 4),
            reason=reason,
            source=source,
        ))

    # Sort by similarity descending, take top_k.
    matches.sort(key=lambda m: m.similarity, reverse=True)
    return matches[:top_k]
