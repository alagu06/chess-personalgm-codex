"""Opening Expert specialist."""

from __future__ import annotations

import html
import re
from collections.abc import Iterator

from langchain_core.messages import HumanMessage, SystemMessage

from src.board_render import fen_with_fallback, render_board_svg
from src.llm_client import get_chat_model

SYSTEM_PROMPT = """You are an Opening Expert — a chess master with encyclopedic knowledge of every
mainline opening, sub-variation, and modern theoretical trend up to your training
cutoff. Your students range from 1200 to 2200 ELO. They want to understand WHY
moves are played, not memorize a tree of variations.

When asked about an opening (by name, ECO code, or move order), produce a
structured lesson with these sections in this order:

1. HEADER — Opening name, ECO code(s), defining move order in algebraic notation.

2. ONE-LINE ESSENCE — What is this opening fundamentally trying to achieve?
   Strip it to one sentence.

3. PAWN STRUCTURE — The typical structure that results. Locked? Open? Asymmetric?
   What does that structure dictate about both sides' plans?

4. STRATEGIC IDEAS — WHITE — 3 to 5 concrete ideas, with exact piece placements
   and pawn breaks. "Develop pieces" is too vague. "Knight to f3 then h4 to harass
   Black's Bf5, supported by g3" is concrete. Use the latter.

5. STRATEGIC IDEAS — BLACK — Same level of specificity for Black's plans.

6. KEY POSITIONS — 2 to 3 important positions in the opening. For each, output:
     a) Move sequence to reach it (e.g. "1.e4 e5 2.Nf3 Nc6 3.Bb5 Nf6 4.O-O Nxe4")
     b) The FEN of the resulting position (compute carefully — do not guess)
     c) A 1-2 line caption explaining what to look at in that position
   Format each one in this exact block so the app can parse and render it as a
   chess board image:

     {{position}}
       moves: 1.e4 e5 2.Nf3 ...
       fen: <FEN string>
       caption: "Starting position — note Black's c8 bishop still inside the chain"
     {{/position}}

7. FAMOUS PRACTITIONERS — 2 to 3 elite players associated with the opening, with
   one-line context (e.g. "Kramnik popularized the Berlin Wall to dethrone
   Kasparov in 2000").

8. AMATEUR vs GM PERCEPTION — How is this opening seen at sub-2000 level vs at
   the elite? They often differ.

9. COMMON PITFALLS — 2 to 3 specific mistakes students make in this opening.
   Be concrete. "Don't trade the light-squared bishop too early" is concrete.
   "Be careful" is not.

10. ONE-SENTENCE SUMMARY — Distill the whole opening into one memorable sentence.

STYLE RULES:
- Use exact piece-and-square notation (Bf5, Nf3, c5-break) — never "the bishop".
- Acknowledge uncertainty. If a line is contested or your training data may be
  stale, say "the consensus through 2024 was X" rather than asserting.
- Don't moralize ("this is a great opening"). Give honest assessments including
  weaknesses, drawishness, complexity for the target rating range.
- Keep paragraphs 3-5 sentences. Bullets where they help.
- Do NOT invent specific theoretical novelties or 15-deep move orders unless you
  are confident they are mainline. When unsure, point to the typical move and let
  the user explore further.

OUTPUT:
- Markdown formatted, with the {{position}}...{{/position}} blocks for any
  positions you reference. The app will render those blocks as SVG board images.
- Concise but complete. Aim for 600-1000 words total.
"""

POSITION_BLOCK_RE = re.compile(
    r"\{\{position\}\}(.*?)\{\{/position\}\}",
    re.DOTALL | re.IGNORECASE,
)


def _strip_quotes(value: str) -> str:
    clean = value.strip()
    if len(clean) >= 2 and clean[0] == clean[-1] and clean[0] in {"'", '"'}:
        return clean[1:-1].strip()
    return clean


def _extract_key(block: str, key: str) -> str:
    match = re.search(
        rf"^\s*{re.escape(key)}\s*:\s*(.+?)\s*$",
        block,
        re.MULTILINE | re.IGNORECASE,
    )
    return _strip_quotes(match.group(1)) if match else ""


def parse_position_blocks(markdown: str) -> list[dict]:
    """Extract position blocks from raw LLM markdown."""

    positions = []
    for match in POSITION_BLOCK_RE.finditer(markdown):
        raw = match.group(0)
        body = match.group(1)
        positions.append(
            {
                "raw": raw,
                "moves": _extract_key(body, "moves"),
                "fen": _extract_key(body, "fen"),
                "caption": _extract_key(body, "caption"),
            }
        )
    return positions


def _messages(opening_name: str, mode: str) -> list:
    return [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Teach this opening or move order: {opening_name}\n"
                f"Teaching mode: {mode}"
            )
        ),
    ]


def teach_opening(opening_name: str, mode: str = "general") -> dict:
    """Blocking version. Returns markdown plus parsed/rendered positions."""

    model = get_chat_model(temperature=0.2)
    response = model.invoke(_messages(opening_name, mode))
    markdown = response.content if isinstance(response.content, str) else str(response.content)
    positions = _positions_with_svgs(markdown)
    return {"markdown": markdown, "positions": positions}


def stream_opening(opening_name: str, mode: str = "general") -> Iterator[str]:
    """Streaming version. Yields markdown chunks as the LLM produces them."""

    model = get_chat_model(temperature=0.2)
    for chunk in model.stream(_messages(opening_name, mode)):
        content = chunk.content
        if isinstance(content, str):
            yield content
        elif content:
            yield str(content)


def _positions_with_svgs(markdown: str) -> list[dict]:
    positions = []
    for position in parse_position_blocks(markdown):
        fen = fen_with_fallback(position["fen"], position["moves"])
        svg = render_board_svg(fen) if fen else None
        enriched = dict(position)
        enriched["fen"] = fen or position["fen"]
        enriched["svg"] = svg
        positions.append(enriched)
    return positions


def render_lesson_html(lesson: dict) -> str:
    """Replace position blocks with centred SVG boards and captions."""

    markdown = lesson.get("markdown", "")
    positions = lesson.get("positions") or _positions_with_svgs(markdown)
    rendered = markdown

    for position in positions:
        raw = position.get("raw", "")
        svg = position.get("svg")
        caption = html.escape(position.get("caption", ""))
        if svg:
            replacement = (
                '<div class="board-wrap">'
                f"{svg}"
                f'<div class="board-caption">{caption}</div>'
                "</div>"
            )
        else:
            replacement = (
                '<div class="board-note">'
                "Position diagram unavailable: the FEN and moves could not be parsed."
                "</div>"
            )
        rendered = rendered.replace(raw, replacement, 1)

    return rendered
