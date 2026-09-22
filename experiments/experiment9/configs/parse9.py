"""
D-mode section parser: splits on the fixed "### A<n>_NAME" header format
requested by prompts8.py. Never guesses boundaries from prose -- a
section that is missing its header is simply absent (OMITTED at the
artifact level), not inferred.
"""
import re

HEADER_RE = re.compile(r"^\s*#{1,4}\s*(A\d_[A-Z_]+)\s*$", re.MULTILINE)


def parse_sections(text, expected_artifact_ids):
    """Returns {artifact_id: section_text_or_None}. Also returns
    parse_quality: FULL (all expected present, in order, non-empty),
    PARTIAL (some present), NONE (no recognized headers at all)."""
    matches = list(HEADER_RE.finditer(text))
    sections = {aid: None for aid in expected_artifact_ids}
    found_ids = []
    for i, m in enumerate(matches):
        aid = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if aid in sections:
            if sections[aid] is None or (body and len(body) > len(sections[aid] or "")):
                sections[aid] = body if body else sections[aid]
            found_ids.append(aid)

    n_present = sum(1 for v in sections.values() if v)
    if n_present == len(expected_artifact_ids):
        quality = "FULL"
    elif n_present == 0:
        quality = "NONE"
    else:
        quality = "PARTIAL"

    return sections, quality, found_ids
