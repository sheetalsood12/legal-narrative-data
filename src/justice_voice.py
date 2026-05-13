"""Split a court opinion into majority, concurrence, and dissent sections."""
import re
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class OpinionSection:
    voice: str
    author: str
    text: str
    start_offset: int
    end_offset: int
    confidence: float


def _find_all_section_headers(text: str) -> List[Dict]:
    matches: List[Dict] = []

    dissent_patterns = [
        r"JUSTICE\s+([A-Z][A-Z]+)\s*,\s*(?:with\s+whom[^,]+,\s*)?dissenting",
        r"CHIEF\s+JUSTICE\s+([A-Z][A-Z]+)\s*,\s*(?:with\s+whom[^,]+,\s*)?dissenting",
    ]
    for pattern in dissent_patterns:
        for m in re.finditer(pattern, text):
            matches.append({
                "start": m.start(), "end": m.end(),
                "author": m.group(1).strip(), "voice": "dissent",
                "header_text": m.group(0),
            })

    concurrence_patterns = [
        r"JUSTICE\s+([A-Z][A-Z]+)\s*,\s*(?:with\s+whom[^,]+,\s*)?concurring(?:\s+in[^.\n]{0,40})?",
        r"CHIEF\s+JUSTICE\s+([A-Z][A-Z]+)\s*,\s*(?:with\s+whom[^,]+,\s*)?concurring(?:\s+in[^.\n]{0,40})?",
    ]
    for pattern in concurrence_patterns:
        for m in re.finditer(pattern, text):
            overlaps_dissent = any(
                d["start"] <= m.start() < d["end"] or m.start() <= d["start"] < m.end()
                for d in matches if d["voice"] == "dissent"
            )
            if overlaps_dissent:
                continue
            matches.append({
                "start": m.start(), "end": m.end(),
                "author": m.group(1).strip(), "voice": "concurrence",
                "header_text": m.group(0),
            })

    matches.sort(key=lambda m: m["start"])

    # Deduplicate close repeats from headers showing up across pages.
    deduped: List[Dict] = []
    for m in matches:
        is_dup = False
        for prev in deduped:
            if (prev["author"] == m["author"]
                    and prev["voice"] == m["voice"]
                    and abs(m["start"] - prev["start"]) < 200):
                is_dup = True
                break
        if not is_dup:
            deduped.append(m)

    return deduped


def detect_sections(opinion_text: str) -> List[OpinionSection]:
    headers = _find_all_section_headers(opinion_text)

    if not headers:
        return [OpinionSection(
            voice="majority", author="UNKNOWN", text=opinion_text,
            start_offset=0, end_offset=len(opinion_text), confidence=0.3,
        )]

    sections: List[OpinionSection] = []
    first = headers[0]
    if first["start"] > 1000:
        sections.append(OpinionSection(
            voice="majority", author="COURT",
            text=opinion_text[:first["start"]],
            start_offset=0, end_offset=first["start"], confidence=0.9,
        ))

    for i, m in enumerate(headers):
        end = headers[i + 1]["start"] if i + 1 < len(headers) else len(opinion_text)
        text = opinion_text[m["start"]:end]
        if len(text) < 200:
            continue
        sections.append(OpinionSection(
            voice=m["voice"], author=m["author"], text=text,
            start_offset=m["start"], end_offset=end, confidence=0.85,
        ))

    return sections


def voice_summary(sections: List[OpinionSection]) -> List[Dict]:
    return [
        {
            "voice": s.voice,
            "author": s.author,
            "char_count": len(s.text),
            "confidence": s.confidence,
            "preview": s.text[:200].replace("\n", " ").strip() + "…",
        }
        for s in sections
    ]


def section_texts_by_voice(sections: List[OpinionSection]) -> Dict[str, str]:
    grouped: Dict[str, List[str]] = {"majority": [], "concurrence": [], "dissent": []}
    for s in sections:
        if s.voice in grouped:
            grouped[s.voice].append(s.text)
    return {voice: "\n\n".join(parts) for voice, parts in grouped.items() if parts}
