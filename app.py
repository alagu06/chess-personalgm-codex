"""Streamlit entry point for PersonalGM."""

from __future__ import annotations

import os
import re
import traceback
from datetime import datetime, timezone

import streamlit as st
from dotenv import load_dotenv

from src.cache import cache_get, cache_set, normalize_key
from src.chess_tools import StockfishUnavailableError
from src.lichess_client import LichessUserNotFoundError
from src.opening_expert import render_lesson_html, stream_opening
from src.state import new_state
from src.supervisor import get_graph

load_dotenv()

st.set_page_config(
    page_title="PersonalGM",
    layout="wide",
    page_icon="♟️",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .hero {
        background: linear-gradient(135deg, #0EA5E9, #38BDF8);
        color: white;
        border-radius: 14px;
        padding: 1.6rem 1.8rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 12px 30px rgba(14, 165, 233, 0.22);
    }
    .hero h1 { margin: 0; font-size: 2.1rem; letter-spacing: 0; }
    .hero p { margin: .35rem 0 0; opacity: .95; }
    [data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(180deg, #E0F2FE, #F0F9FF);
    }
    .board-wrap { text-align: center; margin: 1.2rem auto; }
    .board-wrap svg { max-width: 100%; height: auto; }
    .board-caption { color: #475569; font-size: .92rem; margin-top: .4rem; }
    .board-note {
        border-left: 4px solid #F59E0B;
        padding: .65rem .85rem;
        background: #FFFBEB;
        color: #92400E;
        margin: 1rem 0;
    }
    .cache-badge {
        display: inline-block;
        background: #FEF3C7;
        color: #92400E;
        border-radius: 999px;
        padding: .15rem .55rem;
        font-size: .8rem;
        margin-bottom: .5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _init_session() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("last_trace", [])
    st.session_state.setdefault("phase2_notice", "")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _opening_trace(user_text: str) -> list[dict]:
    return [
        {
            "ts": _now_iso(),
            "agent": "supervisor",
            "tool": "node",
            "args": {"intent": "ask_opening"},
            "result": "route=ask_opening",
            "status": "ok",
        },
        {
            "ts": _now_iso(),
            "agent": "opening_expert",
            "tool": "stream_opening",
            "args": {"opening": user_text[:80]},
            "result": "streamed lesson",
            "status": "ok",
        },
    ]


def classify_intent(text: str) -> str:
    """Tiny rule-based chat intent classifier."""

    lowered = text.lower()
    if "lichess.org/" in lowered or "[event " in lowered:
        return "analyze_game"
    if lowered.startswith("scout "):
        return "scout"
    if "repertoire" in lowered:
        return "repertoire"

    opening_words = (
        "explain",
        "teach me about",
        "what is",
        "show me",
        "opening",
        "gambit",
        "defense",
        "defence",
        "sicilian",
        "caro-kann",
        "french",
        "ruy lopez",
        "queen's gambit",
        "king's indian",
        "nimzo",
    )
    if re.search(r"\b[ABCDE]\d{2}\b", text.upper()):
        return "ask_opening"
    if any(word in lowered for word in opening_words):
        return "ask_opening"
    return "ask_opening"


def _display_error(exc: Exception) -> None:
    if isinstance(exc, LichessUserNotFoundError):
        st.error(str(exc))
        return
    if isinstance(exc, StockfishUnavailableError):
        st.error(str(exc))
        return
    st.error(str(exc))
    if st.session_state.get("dev_mode"):
        st.code(traceback.format_exc(), language="text")


def _append_assistant(content: str, unsafe_html: bool = False) -> None:
    st.session_state.messages.append(
        {"role": "assistant", "content": content or "", "unsafe_html": unsafe_html}
    )


def _run_graph_state(state: dict, unsafe_html: bool = False) -> bool:
    try:
        result = get_graph().invoke(state)
        st.session_state.last_trace = result.get("trace_log", [])
        final_text = result.get("final_text", "")
        _append_assistant(final_text, unsafe_html=unsafe_html)
        return True
    except Exception as exc:
        _display_error(exc)
        return False


def _opening_cache_key(text: str, mode: str = "general") -> str:
    return f"{mode}__{normalize_key(text)}"


@st.dialog("Repertoire Doctor")
def repertoire_dialog() -> None:
    """Collect Lichess username and dispatch the repertoire graph route."""

    with st.form("repertoire_form"):
        username = st.text_input("Lichess username")
        n_games = st.slider("Games to inspect", min_value=10, max_value=200, value=50, step=10)
        submitted = st.form_submit_button("Analyze")

    if submitted:
        if not username.strip():
            st.error("Enter a Lichess username.")
            return
        user_text = f"Analyze repertoire for {username.strip()}"
        cached_analysis = cache_get("repertoire", f"{username.strip().lower()}__{n_games}")
        st.session_state.messages.append({"role": "user", "content": user_text})
        success = _run_graph_state(
            new_state(
                user_message=user_text,
                intent="repertoire",
                username=username.strip(),
                repertoire_data={"n_games": n_games},
            )
        )
        if success and cached_analysis is not None:
            st.session_state.messages[-1]["content"] = (
                "⚡ cached\n\n" + st.session_state.messages[-1]["content"]
            )
        if success:
            st.rerun()


@st.dialog("Opponent Scout")
def scout_dialog() -> None:
    """Collect opponent username and dispatch Scout."""

    with st.form("scout_form"):
        username = st.text_input("Opponent Lichess username")
        submitted = st.form_submit_button("Scout")

    if submitted:
        if not username.strip():
            st.error("Enter a Lichess username.")
            return
        user_text = f"scout {username.strip()}"
        st.session_state.messages.append({"role": "user", "content": user_text})
        if _run_graph_state(new_state(user_message=user_text, intent="scout", username=username.strip())):
            st.rerun()


@st.dialog("Postgame Analyst")
def postgame_dialog() -> None:
    """Collect a Lichess URL or raw PGN and dispatch Postgame Analyst."""

    with st.form("postgame_form"):
        game_input = st.text_area("Lichess game URL or raw PGN", height=180)
        submitted = st.form_submit_button("Analyze")

    if submitted:
        if not game_input.strip():
            st.error("Paste a Lichess game URL or PGN.")
            return
        st.session_state.messages.append({"role": "user", "content": "Analyze pasted game"})
        if _run_graph_state(new_state(user_message=game_input.strip(), intent="analyze_game")):
            st.rerun()


def render_sidebar() -> None:
    """Render the application sidebar."""

    with st.sidebar:
        st.markdown("## ♟️ PersonalGM")
        st.divider()
        st.markdown("### Features")
        if st.button("📊 Repertoire Doctor", use_container_width=True):
            repertoire_dialog()
        if st.button("⚔️ Opponent Scout", use_container_width=True):
            scout_dialog()
        if st.button("🔍 Postgame Analyst", use_container_width=True):
            postgame_dialog()

        st.divider()
        st.markdown("### Settings")
        st.toggle("Dev mode", key="dev_mode")
        if st.button("🧹 Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_trace = []
            st.rerun()

        st.divider()
        model_name = os.getenv("CHESS_LLM_MODEL", "gemini-2.5-flash")
        st.caption(f"LLM: Gemini · {model_name}")


def render_main() -> None:
    """Render hero, trace, chat history, and chat input."""

    st.markdown(
        """
        <div class="hero">
          <h1>♟️ PersonalGM — Your AI Chess Coach</h1>
          <p>Opening lessons and repertoire diagnosis for ambitious club players.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.get("dev_mode"):
        with st.expander("🛠 Trace", expanded=False):
            trace = st.session_state.get("last_trace", [])
            if trace:
                st.table(trace)
            else:
                st.caption("No trace yet.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message.get("unsafe_html"):
                st.markdown(message["content"], unsafe_allow_html=True)
            else:
                st.markdown(message["content"])

    text = st.chat_input("Ask anything...")
    if not text:
        return

    st.session_state.messages.append({"role": "user", "content": text})
    with st.chat_message("user"):
        st.markdown(text)

    intent = classify_intent(text)
    if intent == "ask_opening":
        with st.chat_message("assistant"):
            placeholder = st.empty()
            try:
                key = _opening_cache_key(text)
                cached_lesson = cache_get("openings", key)
                if cached_lesson:
                    st.markdown('<span class="cache-badge">⚡ cached</span>', unsafe_allow_html=True)
                    lesson = cached_lesson
                else:
                    raw_markdown = placeholder.write_stream(stream_opening(text)) or ""
                    lesson = {"markdown": raw_markdown, "positions": []}
                    cache_set("openings", key, lesson)
                rendered_html = render_lesson_html(lesson)
                placeholder.markdown(rendered_html, unsafe_allow_html=True)
                st.session_state.last_trace = _opening_trace(text)
                _append_assistant(rendered_html, unsafe_html=True)
            except Exception as exc:
                _display_error(exc)
    elif intent == "repertoire":
        with st.chat_message("assistant"):
            guidance = "Use the Repertoire Doctor button in the sidebar and enter your Lichess username."
            st.info(guidance)
            _append_assistant(guidance)
    elif intent == "scout":
        username = text.split(maxsplit=1)[1].strip() if len(text.split(maxsplit=1)) > 1 else ""
        with st.chat_message("assistant"):
            if username:
                success = _run_graph_state(new_state(user_message=text, intent="scout", username=username))
                if success:
                    st.markdown(st.session_state.messages[-1]["content"])
            else:
                guidance = "Type `scout lichess_username` or use the Opponent Scout button in the sidebar."
                st.info(guidance)
                _append_assistant(guidance)
    elif intent == "analyze_game":
        with st.chat_message("assistant"):
            success = _run_graph_state(new_state(user_message=text, intent="analyze_game"))
            if success:
                st.markdown(st.session_state.messages[-1]["content"])


def main() -> None:
    """Run the Streamlit app."""

    _init_session()
    render_sidebar()
    render_main()


if __name__ == "__main__":
    main()
