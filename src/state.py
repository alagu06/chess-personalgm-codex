"""Shared LangGraph state for PersonalGM."""

from __future__ import annotations

import operator
from typing import Annotated, Optional, TypedDict


class TraceEntry(TypedDict, total=False):
    """Small JSON-serialisable execution trace entry."""

    ts: str
    agent: str
    tool: str
    args: dict
    result: str
    status: str


class PersonalGMState(TypedDict, total=False):
    """State passed through the PersonalGM LangGraph supervisor."""

    user_message: str
    intent: str
    username: Optional[str]
    current_opening: Optional[str]
    teach_mode: str
    repertoire_data: Optional[dict]
    opening_lesson: Optional[dict]
    final_text: str
    trace_log: Annotated[list[TraceEntry], operator.add]


VALID_INTENTS = (
    "ask_opening",
    "repertoire",
    "scout",
    "analyze_game",
)


def new_state(**overrides) -> PersonalGMState:
    """Create a state object with sane defaults."""

    base: PersonalGMState = {
        "user_message": "",
        "intent": "ask_opening",
        "teach_mode": "general",
        "trace_log": [],
        "final_text": "",
    }
    base.update(overrides)
    return base
