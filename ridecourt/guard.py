"""Guard — validates agent output against the evidence pack and policy.

``check(output, evidence_pack, policy_ids) -> list[Problem]``

Three checks:

1. **Citation check**: every cite in agent text (``E#`` or ``P-..``) must
   exist in the evidence pack or the policy clause list.

2. **Number check**: every number that appears in the agent's text fields
   (money like ``S$3.25``, distances like ``3.25 km``, durations like
   ``5 min``, metres like ``150 m``, percentages like ``15%``) must match a
   number found in the **cited** evidence item's ``data`` dict or the policy
   engine's ``computation`` string — *with the same unit*.  ``S$3.25`` must
   not be backed by ``3.25 km`` in the evidence.

3. **Forbidden-words check**: no promise or certainty words the policy does
   not allow (``"guaranteed"``, ``"definitely fraud"``, etc.).

The orchestrator re-asks the agent once with the problems listed, then
escalates to a human.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Problem
# ---------------------------------------------------------------------------


@dataclass
class Problem:
    """A single guard violation."""

    kind: str  # "unknown_cite" | "unverified_number" | "forbidden_word"
    detail: str
    cite: str = ""
    field: str = ""


# ---------------------------------------------------------------------------
# Forbidden words
# ---------------------------------------------------------------------------

FORBIDDEN_WORDS: frozenset[str] = frozenset(
    {
        "guaranteed",
        "definitely fraud",
        "100% certain",
        "absolutely sure",
        "i promise",
        "i guarantee",
        "definitely a scam",
        "certainly fraud",
        "without a doubt fraud",
    }
)


# ---------------------------------------------------------------------------
# Number patterns (unit-aware)
# ---------------------------------------------------------------------------

# Money: S$3.25, $3.25, SGD 3.25
_MONEY_RE = re.compile(r"(?:S\$|SGD\s*|\$)\s*(\d+(?:\.\d+)?)", re.IGNORECASE)

# Kilometres: 3.25 km, 3.25km
_KM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*km\b", re.IGNORECASE)

# Metres: 150 m, 150m (but NOT "km" — so negative lookahead)
_M_RE = re.compile(r"(\d+(?:\.\d+)?)\s*m\b(?!in\b|s\b)", re.IGNORECASE)

# Minutes: 5 min, 5min, 5 minutes
_MIN_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:min|minutes)\b", re.IGNORECASE)

# Percent: 15%, 15.5 %
_PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%", re.IGNORECASE)


@dataclass
class ExtractedNumber:
    value: float
    unit: str  # "money" | "km" | "m" | "min" | "percent"
    raw: str
    field: str = ""


# ---------------------------------------------------------------------------
# Citation patterns
# ---------------------------------------------------------------------------

# E1, E12, etc.
_CITE_E_RE = re.compile(r"\bE(\d+)\b")
# P-RD-1, P-NS-2, etc.
_CITE_P_RE = re.compile(r"\bP-[A-Z]+-\d+\b")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def check(
    output: BaseModel,
    evidence_pack: Any | None,
    policy_ids: Sequence[str] = (),
    *,
    policy_result: Any | None = None,
) -> list[Problem]:
    """Validate *output* against the evidence pack and policy clause list.

    Parameters
    ----------
    output
        A pydantic model instance returned by an agent (e.g.
        ``RiderAdvocateOutput``).
    evidence_pack
        ``EvidencePack`` with items ``E1..En``.
    policy_ids
        List of valid policy clause IDs for this case (e.g.
        ``["P-RD-1", "P-NS-2"]``).
    policy_result
        Optional ``PolicyResult`` — its ``computation`` string and ``amount``
        are also valid number sources (the policy engine is deterministic).
    """
    problems: list[Problem] = []

    valid_eids = _evidence_ids(evidence_pack)
    valid_pids = set(policy_ids)
    # Also include policy_result's applicable_clauses.
    if policy_result is not None:
        for c in getattr(policy_result, "applicable_clauses", []) or []:
            valid_pids.add(c)

    # Collect all text fields from the output.
    texts = _collect_text(output)
    # Also collect any field named "cites" or lists of cites.
    cite_fields = _collect_cites(output)

    # --- 1. Citation check ---
    for field_name, text in texts:
        for cite in _find_cites(text):
            if not _cite_exists(cite, valid_eids, valid_pids):
                problems.append(
                    Problem(
                        kind="unknown_cite",
                        detail=f"Citation {cite} not found in evidence pack or policy clauses.",
                        cite=cite,
                        field=field_name,
                    )
                )
    for field_name, cites in cite_fields:
        for cite in cites:
            if not _cite_exists(cite, valid_eids, valid_pids):
                problems.append(
                    Problem(
                        kind="unknown_cite",
                        detail=f"Citation {cite} not found in evidence pack or policy clauses.",
                        cite=cite,
                        field=field_name,
                    )
                )

    # --- 2. Number check ---
    # Build the set of valid (value, unit) pairs from cited evidence.
    cited_eids = _cited_eids(texts, cite_fields)
    valid_numbers = _valid_numbers(evidence_pack, cited_eids, policy_result)

    for field_name, text in texts:
        extracted = _extract_numbers(text, field_name)
        for num in extracted:
            if not _number_verified(num, valid_numbers):
                problems.append(
                    Problem(
                        kind="unverified_number",
                        detail=(
                            f"Number {num.raw} ({num.value} {num.unit}) in "
                            f"field '{field_name}' does not match any cited "
                            "evidence or policy computation value with the "
                            "same unit."
                        ),
                        field=field_name,
                    )
                )

    # --- 3. Forbidden-words check ---
    for field_name, text in texts:
        low = text.lower()
        for word in FORBIDDEN_WORDS:
            if word in low:
                problems.append(
                    Problem(
                        kind="forbidden_word",
                        detail=f"Forbidden certainty word '{word}' in field '{field_name}'.",
                        field=field_name,
                    )
                )

    return problems


# ---------------------------------------------------------------------------
# Helpers: evidence IDs
# ---------------------------------------------------------------------------


def _evidence_ids(pack: Any | None) -> set[str]:
    if pack is None:
        return set()
    items = getattr(pack, "items", None)
    if items is None:
        return set()
    return {getattr(i, "id", "") for i in items if getattr(i, "id", "")}


def _cite_exists(cite: str, valid_eids: set[str], valid_pids: set[str]) -> bool:
    return cite in valid_eids or cite in valid_pids


# ---------------------------------------------------------------------------
# Helpers: collect text and cites from a pydantic model
# ---------------------------------------------------------------------------


def _collect_text(model: BaseModel) -> list[tuple[str, str]]:
    """Yield (field_name, text) for every string field (recursively)."""
    texts: list[tuple[str, str]] = []
    _walk_text(model, "", texts)
    return texts


def _walk_text(obj: Any, path: str, texts: list[tuple[str, str]]) -> None:
    if isinstance(obj, BaseModel):
        for name, value in obj.__dict__.items():
            _walk_text(value, f"{path}.{name}" if path else name, texts)
    elif isinstance(obj, str):
        texts.append((path, obj))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            _walk_text(item, f"{path}[{i}]", texts)


def _collect_cites(model: BaseModel) -> list[tuple[str, list[str]]]:
    """Collect explicit ``cites`` fields (lists of E#/P-# strings)."""
    result: list[tuple[str, list[str]]] = []
    _walk_cites(model, "", result)
    return result


def _walk_cites(obj: Any, path: str, result: list[tuple[str, list[str]]]) -> None:
    if isinstance(obj, BaseModel):
        for name, value in obj.__dict__.items():
            _walk_cites(value, f"{path}.{name}" if path else name, result)
    elif isinstance(obj, list):
        for item in obj:
            _walk_cites(item, path, result)
    elif isinstance(obj, str) and path.split(".")[-1] == "cites":
        # Only collects if the field is literally named 'cites' and is a list
        # of strings — handled by the list branch above.
        pass


# Re-do _collect_cites to handle the common pattern: a field named "cites"
# that is a list[str], and lists of objects each with a "cites" list[str].


def _collect_cites(model: BaseModel) -> list[tuple[str, list[str]]]:
    """Collect explicit ``cites`` fields (lists of E#/P-# strings)."""
    result: list[tuple[str, list[str]]] = []
    _walk_for_cites(model, "", result)
    return result


def _walk_for_cites(obj: Any, path: str, result: list[tuple[str, list[str]]]) -> None:
    if isinstance(obj, BaseModel):
        for name, value in obj.__dict__.items():
            child_path = f"{path}.{name}" if path else name
            if name == "cites" and isinstance(value, list):
                cites = [
                    v for v in value if isinstance(v, str)
                ]
                if cites:
                    result.append((child_path, cites))
            else:
                _walk_for_cites(value, child_path, result)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            _walk_for_cites(item, path, result)


# ---------------------------------------------------------------------------
# Helpers: find citations in text
# ---------------------------------------------------------------------------


def _find_cites(text: str) -> list[str]:
    cites = []
    cites.extend(m.group(0) for m in _CITE_E_RE.finditer(text))
    cites.extend(m.group(0) for m in _CITE_P_RE.finditer(text))
    return cites


def _cited_eids(texts: list[tuple[str, str]], cite_fields: list[tuple[str, list[str]]]) -> set[str]:
    """Collect E# IDs cited anywhere in the output text or cite lists."""
    eids: set[str] = set()
    for _, text in texts:
        for m in _CITE_E_RE.finditer(text):
            eids.add(m.group(0))
    for _, cites in cite_fields:
        for cite in cites:
            if _CITE_E_RE.fullmatch(cite):
                eids.add(cite)
    return eids


# ---------------------------------------------------------------------------
# Helpers: valid numbers from cited evidence + policy result
# ---------------------------------------------------------------------------

_TOLERANCE = 0.01  # 1 cent / 0.01 unit


def _valid_numbers(
    evidence_pack: Any | None,
    cited_eids: set[str],
    policy_result: Any | None,
) -> list[tuple[float, str]]:
    """Build a list of (value, unit) pairs that the guard will accept."""
    pairs: list[tuple[float, str]] = []
    if evidence_pack is not None:
        for item in getattr(evidence_pack, "items", []):
            eid = getattr(item, "id", "")
            # If the agent cited specific EIDs, only use those. Otherwise use all.
            if cited_eids and eid not in cited_eids:
                continue
            data = getattr(item, "data", {})
            for value, unit in _extract_data_numbers(data, item.summary):
                pairs.append((value, unit))

    # Policy result computation string — extract numbers with units.
    if policy_result is not None:
        comp = getattr(policy_result, "computation", "") or ""
        for num in _extract_numbers(comp, ""):
            pairs.append((num.value, num.unit))
        amount = getattr(policy_result, "amount", None)
        if amount is not None:
            pairs.append((float(amount), "money"))

    return pairs


def _extract_data_numbers(data: dict, summary: str) -> list[tuple[float, str]]:
    """Extract (value, unit) from an evidence item's data dict.

    The unit is inferred from the key name:
    ``_km`` → km, ``_m`` → m (metres), ``_min`` → min,
    ``fare``, ``charged``, ``total``, ``amount`` → money,
    ``pct`` / ``percent`` → percent.
    """
    pairs: list[tuple[float, str]] = []
    for key, val in data.items():
        if isinstance(val, bool):
            continue
        if isinstance(val, (int, float)):
            unit = _unit_from_key(key)
            pairs.append((float(val), unit))
        elif isinstance(val, dict):
            pairs.extend(_extract_data_numbers(val, ""))
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, dict):
                    pairs.extend(_extract_data_numbers(item, ""))
                elif isinstance(item, (int, float)):
                    pairs.append((float(item), "count"))
    # Also extract numbers from the summary text (they have units).
    for num in _extract_numbers(summary, ""):
        pairs.append((num.value, num.unit))
    return pairs


def _unit_from_key(key: str) -> str:
    low = key.lower()
    if low.endswith("_km"):
        return "km"
    if low.endswith("_m") or "distance_to_pickup" in low or "_metres" in low:
        return "m"
    if low.endswith("_min") or "minutes" in low:
        return "min"
    if "pct" in low or "percent" in low or low.endswith("_pct"):
        return "percent"
    if any(w in low for w in ("fare", "charged", "total", "amount", "fee", "credit", "discount")):
        return "money"
    if "speed" in low:
        return "kmh"
    if "rating" in low:
        return "rating"
    if "count" in low or low.endswith("_90d") or "attempts" in low:
        return "count"
    if "age" in low or "days" in low:
        return "count"
    return "count"


# ---------------------------------------------------------------------------
# Helpers: extract numbers from agent text
# ---------------------------------------------------------------------------


def _extract_numbers(text: str, field_name: str) -> list[ExtractedNumber]:
    result: list[ExtractedNumber] = []

    for rx, unit in (
        (_MONEY_RE, "money"),
        (_KM_RE, "km"),
        (_MIN_RE, "min"),
        (_PCT_RE, "percent"),
        (_M_RE, "m"),
    ):
        for m in rx.finditer(text):
            raw = m.group(0)
            value = float(m.group(1))
            result.append(ExtractedNumber(value, unit, raw, field_name))

    return result


def _number_verified(num: ExtractedNumber, valid: list[tuple[float, str]]) -> bool:
    """Check whether *num* matches some valid (value, unit) pair within
    tolerance and with the same unit."""
    for value, unit in valid:
        if unit != num.unit:
            continue
        if abs(value - num.value) <= _TOLERANCE:
            return True
    return False
