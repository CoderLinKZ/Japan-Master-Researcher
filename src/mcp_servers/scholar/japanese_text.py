"""Narrow Japanese university-name spelling normalization for search matching."""

from __future__ import annotations

import unicodedata

# These are common input variants, not a full Simplified-Chinese conversion.
_JAPANESE_KANJI_VARIANTS = str.maketrans(
    {
        "稻": "稲",
        "干": "幹",
        "报": "報",
        "术": "術",
        "东": "東",
        "电": "電",
        "庆": "慶",
        "馆": "館",
    }
)


def to_japanese_spelling(value: str) -> str:
    """Translate a bounded set of Japanese-school kanji variants."""

    return value.translate(_JAPANESE_KANJI_VARIANTS)


def normalize_search_text(value: str) -> str:
    """Fold width, common kanji variants, case, and punctuation for matching."""

    normalized = to_japanese_spelling(unicodedata.normalize("NFKC", value)).casefold()
    return "".join(character for character in normalized if character.isalnum())
