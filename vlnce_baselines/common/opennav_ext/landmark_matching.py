import re
from typing import Any, Iterable, List, Optional, Sequence


_DEFAULT_EQUIVALENTS = {
    "entry way": ("entryway",),
    "entryway": ("entry way",),
}


def normalize_text(text: Any) -> str:
    """Normalize text for phrase-boundary matching."""
    lowered = str(text or "").strip().lower().replace("_", " ")
    return re.sub(r"\s+", " ", lowered)


def token_sequence(text: Any) -> List[str]:
    return re.findall(r"[a-z0-9]+", normalize_text(text))


def unique_normalized(values: Iterable[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for value in values:
        normalized = normalize_text(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(str(value).strip())
    return result


def split_landmark_terms(landmarks: Any) -> List[str]:
    chunks = re.split(r"[,;\n]+", str(landmarks or ""))
    cleaned = []
    for chunk in chunks:
        text = re.sub(r"^\s*[-*\d.)]+\s*", "", chunk).strip()
        text = re.sub(
            r"^\s*(?:action|step|landmark|target|final landmark|final target)\s*\d*\s*[:.)-]\s*",
            "",
            text,
            flags=re.I,
        ).strip()
        if text:
            cleaned.append(text)
    return unique_normalized(cleaned)


def final_landmark_terms(landmarks: Any) -> List[str]:
    """Return the final destination landmark group from a landmark list."""
    terms = split_landmark_terms(landmarks)
    if not terms:
        return []
    final_term = terms[-1]
    alternatives = [
        part.strip()
        for part in re.split(r"\s*(?:/|\||\bor\b)\s*", final_term)
        if part.strip()
    ]
    return unique_normalized(alternatives or [final_term])


def _equivalent_terms(
    term: str,
    equivalents: Optional[dict] = None,
) -> List[str]:
    normalized = normalize_text(term)
    mapping = equivalents if equivalents is not None else _DEFAULT_EQUIVALENTS
    terms = [normalized]
    terms.extend(mapping.get(normalized, ()))
    return unique_normalized(terms)


def _contains_phrase(value_tokens: Sequence[str], phrase_tokens: Sequence[str]) -> bool:
    if not phrase_tokens or len(phrase_tokens) > len(value_tokens):
        return False
    limit = len(value_tokens) - len(phrase_tokens) + 1
    for start in range(limit):
        if list(value_tokens[start : start + len(phrase_tokens)]) == list(phrase_tokens):
            return True
    return False


def term_present(
    term: str,
    values: Iterable[Any],
    equivalents: Optional[dict] = None,
) -> bool:
    """Return True only on exact token / contiguous phrase matches.

    This intentionally avoids substring matches such as door->doorway,
    room->bedroom, and hall->hallway.
    """
    candidate_terms = _equivalent_terms(term, equivalents)
    candidate_token_sequences = [
        token_sequence(candidate_term) for candidate_term in candidate_terms
    ]
    candidate_token_sequences = [
        tokens for tokens in candidate_token_sequences if tokens
    ]
    if not candidate_token_sequences:
        return False

    for value in values:
        value_tokens = token_sequence(value)
        if not value_tokens:
            continue
        for phrase_tokens in candidate_token_sequences:
            if _contains_phrase(value_tokens, phrase_tokens):
                return True
    return False


def matched_terms(
    terms: Iterable[str],
    values: Iterable[Any],
    equivalents: Optional[dict] = None,
) -> List[str]:
    return [
        term
        for term in unique_normalized(terms)
        if term_present(term, values, equivalents=equivalents)
    ]


def missing_terms(
    terms: Iterable[str],
    values: Iterable[Any],
    equivalents: Optional[dict] = None,
) -> List[str]:
    return [
        term
        for term in unique_normalized(terms)
        if not term_present(term, values, equivalents=equivalents)
    ]
