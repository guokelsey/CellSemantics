"""Compatibility label helpers, not the earlier VAE training program."""

import re


def parse_rich_label(text: str):
    """Extract (cell_type, organ) from rich label text."""
    m = re.match(r"This is a (.+?) from the (.+?) of donor", text)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    if "|" in text:
        parts = text.split("|", 1)
        return parts[0].strip(), parts[1].strip()
    return text.strip(), ""


def parse_target(text):
    """Parse 'A | B' format into (A, B)."""
    if "|" in text:
        parts = text.split("|", 1)
        return parts[0].strip(), parts[1].strip()
    return text.strip(), ""


def normalize(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def fuzzy_match(gt: str, pred: str) -> bool:
    """Fuzzy string match: exact or substring in either direction."""
    g, p = normalize(gt), normalize(pred)
    if not g or not p:
        return False
    return g == p or g in p or p in g
