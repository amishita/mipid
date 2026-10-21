
import re


def detect_language(text):
    """Return a simple script-level language clue."""

    has_devanagari = bool(re.search(r"[\u0900-\u097F]", text))
    has_latin = bool(re.search(r"[A-Za-z]", text))

    if has_devanagari and has_latin:
        return "Hindi + Latin script (possible code-switching)"

    if has_devanagari:
        return "Hindi script detected"

    if has_latin:
        return "Latin script detected (language uncertain)"

    return "Unknown"
