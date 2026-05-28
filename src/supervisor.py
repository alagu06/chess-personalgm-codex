"""LangGraph supervisor wiring for PersonalGM."""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache

from src.opening_coach import analyze_repertoire, render_repertoire_markdown
from src.opening_expert import render_lesson_html, teach_opening
from src.state import VALID_INTENTS, PersonalGMState, TraceEntry


def _trace(agent: str, tool: str, args: dict, result: str, status: str = "ok") -> TraceEntry:
    return {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "agent": agent,
        "tool": tool,
        "args": args,
        "result": result,
        "status": status,
    }


def supervisor_node(state: PersonalGMState) -> dict:
    """Normalize intent and log routing."""

    intent = state.get("intent", "ask_opening")
    if intent not in VALID_INTENTS:
        intent = "ask_opening"
    return {
        "intent": intent,
        "trace_log": [
            _trace(
                "supervisor",
                "node",
                {"intent": state.get("intent", "")},
                f"route={intent}",
            )
        ],
    }


def opening_expert_node(state: PersonalGMState) -> dict:
    """Run the Opening Expert specialist."""

    lesson = teach_opening(
        state.get("user_message", ""),
        state.get("teach_mode", "general"),
    )
    return {
        "opening_lesson": lesson,
        "final_text": render_lesson_html(lesson),
        "trace_log": [
            _trace(
                "opening_expert",
                "teach_opening",
                {"opening": state.get("user_message", "")[:80]},
                "lesson generated",
            )
        ],
    }


def opening_coach_node(state: PersonalGMState) -> dict:
    """Run the Repertoire Doctor specialist."""

    username = state.get("username") or state.get("user_message", "")
    input_options = state.get("repertoire_data") or {}
    n_games = int(input_options.get("n_games", 50))
    analysis = analyze_repertoire(username, n_games=n_games)
    return {
        "repertoire_data": analysis,
        "final_text": render_repertoire_markdown(analysis),
        "trace_log": [
            _trace(
                "opening_coach",
                "analyze_repertoire",
                {"username": username, "n_games": n_games},
                "repertoire analyzed",
            )
        ],
    }


def scout_stub_node(state: PersonalGMState) -> dict:
    """Phase 2 placeholder."""

    return {
        "final_text": "⏸ **Opponent Scout** — coming in Phase 2. Not yet implemented.",
        "trace_log": [
            _trace("opponent_scout", "node", {}, "Phase 2 placeholder")
        ],
    }


def postgame_stub_node(state: PersonalGMState) -> dict:
    """Phase 2 placeholder."""

    return {
        "final_text": "⏸ **Postgame Analyst** — coming in Phase 2. Not yet implemented.",
        "trace_log": [
            _trace("postgame_analyst", "node", {}, "Phase 2 placeholder")
        ],
    }


def route_after_supervisor(state: PersonalGMState) -> str:
    """Return the specialist route after supervisor normalization."""

    intent = state.get("intent", "ask_opening")
    return intent if intent in VALID_INTENTS else "ask_opening"


@lru_cache(maxsize=1)
def get_graph():
    """Build and cache the PersonalGM LangGraph."""

    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(PersonalGMState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("ask_opening", opening_expert_node)
    graph.add_node("repertoire", opening_coach_node)
    graph.add_node("scout", scout_stub_node)
    graph.add_node("analyze_game", postgame_stub_node)
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "ask_opening": "ask_opening",
            "repertoire": "repertoire",
            "scout": "scout",
            "analyze_game": "analyze_game",
        },
    )
    for terminal in ("ask_opening", "repertoire", "scout", "analyze_game"):
        graph.add_edge(terminal, END)
    return graph.compile()
