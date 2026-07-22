from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any

try:
    from ftfy import fix_text as ftfy_fix_text
except Exception:  # optional dependency
    ftfy_fix_text = None


MOJIBAKE_MARKERS = ("Ã", "Â", "Ä", "Æ", "áº", "á»", "�")


VIETNAMESE_CHARS = set(
    "àáạảãâầấậẩẫăằắặẳẵ"
    "èéẹẻẽêềếệểễ"
    "ìíịỉĩ"
    "òóọỏõôồốộổỗơờớợởỡ"
    "ùúụủũưừứựửữ"
    "ỳýỵỷỹđ"
    "ÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴ"
    "ÈÉẸẺẼÊỀẾỆỂỄ"
    "ÌÍỊỈĨ"
    "ÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠ"
    "ÙÚỤỦŨƯỪỨỰỬỮ"
    "ỲÝỴỶỸĐ"
)


CHAR_TRANSLATION = str.maketrans(
    {
        "\ufeff": "",
        "\u00ad": "",
        "\u00a0": " ",
        "\u200b": "",
        "\u200c": "",
        "\u200d": "",
        "\u2060": "",
        "ﬀ": "ff",
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
        "−": "-",
        "–": "-",
        "—": "-",
        "‐": "-",
        "“": '"',
        "”": '"',
        "„": '"',
        "’": "'",
        "‘": "'",
        "′": "'",
        "″": '"',
        "×": " x ",
        "÷": " / ",
        "≤": " <= ",
        "≥": " >= ",
        "≦": " <= ",
        "≧": " >= ",
        "±": " +/- ",
        "≈": " ~= ",
        "≠": " != ",
        "→": " -> ",
        "←": " <- ",
        "↔": " <-> ",
        "⇒": " => ",
        "↑": " tang ",
        "↓": " giam ",
        "•": "- ",
        "●": "- ",
        "▪": "- ",
        "◦": "- ",
        "Ƣ": "Ư",
        "ƣ": "ư",
        "µ": "micro",
        "μ": "micro",
        "°": " degree ",
        "℃": " degree C ",
        "℉": " degree F ",
        "‰": " per mille ",
    }
)


CID_REPLACEMENTS = {
    "(cid:54)": ">=",
    "(cid:110)": "↑",
    "(cid:112)": "↓",
    "(cid:113)": "°",
    "(cid:114)": "↔",
}


SUPERSCRIPT_TRANSLATION = str.maketrans(
    {
        "⁰": "^0",
        "¹": "^1",
        "²": "^2",
        "³": "^3",
        "⁴": "^4",
        "⁵": "^5",
        "⁶": "^6",
        "⁷": "^7",
        "⁸": "^8",
        "⁹": "^9",
        "⁺": "^+",
        "⁻": "^-",
        "⁽": "^(",
        "⁾": "^)",
        "ⁿ": "^n",
    }
)


SUBSCRIPT_TRANSLATION = str.maketrans(
    {
        "₀": "0",
        "₁": "1",
        "₂": "2",
        "₃": "3",
        "₄": "4",
        "₅": "5",
        "₆": "6",
        "₇": "7",
        "₈": "8",
        "₉": "9",
        "₊": "+",
        "₋": "-",
        "₍": "(",
        "₎": ")",
    }
)


MEDICAL_UNIT_PATTERNS = (
    (re.compile(r"\bmicro\s*g\b", flags=re.IGNORECASE), "mcg"),
    (re.compile(r"\bmicro\s*l\b", flags=re.IGNORECASE), "uL"),
    (re.compile(r"\bmicro\s*mol\b", flags=re.IGNORECASE), "umol"),
)


def repair_mojibake(text: str) -> str:
    """Repair common UTF-8 text decoded as cp1252/latin1."""
    if not text:
        return ""

    candidates = [text]
    current = text
    for _ in range(2):
        for encoding in ("cp1252", "latin1"):
            try:
                repaired = current.encode(encoding).decode("utf-8")
            except UnicodeError:
                continue
            if repaired not in candidates:
                candidates.append(repaired)
        current = candidates[-1]

    if ftfy_fix_text is not None:
        try:
            fixed = ftfy_fix_text(text)
            if fixed not in candidates:
                candidates.append(fixed)
        except Exception:
            pass

    def score(value: str) -> int:
        marker_penalty = sum(value.count(marker) for marker in MOJIBAKE_MARKERS) * 80
        replacement_penalty = value.count("\ufffd") * 200
        vietnamese_bonus = sum(1 for ch in value if ch in VIETNAMESE_CHARS) * 3
        ascii_bonus = sum(1 for ch in value if ch.isascii()) // 20
        return vietnamese_bonus + ascii_bonus - marker_penalty - replacement_penalty

    return max(candidates, key=score)


def normalize_text(text: str, *, preserve_lines: bool = True) -> str:
    """Normalize PDF text while preserving medical/scientific meaning."""
    if not text:
        return ""

    text = repair_mojibake(text)
    text = unicodedata.normalize("NFC", text)
    for source, replacement in CID_REPLACEMENTS.items():
        text = text.replace(source, replacement)
    text = text.translate(CHAR_TRANSLATION)
    text = text.translate(SUPERSCRIPT_TRANSLATION)
    text = text.translate(SUBSCRIPT_TRANSLATION)

    for pattern, replacement in MEDICAL_UNIT_PATTERNS:
        text = pattern.sub(replacement, text)

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text, flags=re.UNICODE)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)

    if not preserve_lines:
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_for_search(text: str) -> str:
    text = normalize_text(text, preserve_lines=False).lower()
    decomposed = unicodedata.normalize("NFD", text)
    without_marks = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", without_marks)


def stable_id(*parts: Any) -> str:
    raw = "|".join(str(part) for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def safe_filename(value: str) -> str:
    value = normalize_for_search(value)
    value = re.sub(r"[^a-z0-9_.-]+", "_", value)
    return value.strip("._") or "document"
