# PersonalGM — Codex Build Spec

> A multi-agent personal chess coach. You (Codex) are the implementer.
> This document is your specification. Build **Phase 1 only** and stop.

---

## Mission

Build a Streamlit web app called **PersonalGM** — a personal chess coach that helps an amateur player (1200–2200 ELO) improve through four specialist AI agents, orchestrated by a LangGraph supervisor.

The four specialists (final vision):

| # | Specialist        | What it does                                                   | Build phase |
|---|-------------------|----------------------------------------------------------------|-------------|
| 1 | **Opening Expert**     | Free-text Q&A: "explain the Caro-Kann Advance"             | Phase 1 ✅ |
| 2 | **Repertoire Doctor**  | Pulls Lichess games, finds the user's weakest opening      | Phase 1 ✅ |
| 3 | **Opponent Scout**     | Pulls an opponent's games, tells you what they play        | Phase 2 ⏸ |
| 4 | **Postgame Analyst**   | Analyses a single game with Stockfish + LLM commentary     | Phase 2 ⏸ |

In Phase 1, the Opponent Scout and Postgame Analyst sidebar buttons exist but show "Coming in Phase 2 — not yet implemented" placeholders. Their dialogs/forms should NOT be wired.

---

## Tech stack (NO substitutions, NO additions without asking)

- **Python 3.12** (assume Windows; the OS check matters for one cert dep)
- **Streamlit** (UI framework — uses `st.chat_input`, `st.chat_message`, `st.write_stream`)
- **LangGraph** ≥0.2 — supervisor orchestration
- **LangChain** ≥0.3 + `langchain-google-genai` ≥2.0 — LLM abstraction
- **python-chess** ≥1.11 — FEN parsing, SVG board rendering
- **requests** ≥2.31 — Lichess HTTP API
- **python-dotenv** ≥1.0 — `.env` loading
- **pip-system-certs** ≥4.0 (Windows only — corporate proxy compatibility)

Do **NOT** add: `litellm`, `langchain-openai`, `duckdb`, `pandas`, `zstandard`, `pytest` — those are Phase 2 / out of scope.

Do **NOT** add: a database, Redis, Celery, FastAPI, Docker — Phase 1 is a single Streamlit process.

---

## LLM provider — Gemini only in Phase 1

Use `langchain-google-genai`'s `ChatGoogleGenerativeAI`. Read configuration from environment:

| Env var               | Purpose                          | Example                  |
|-----------------------|----------------------------------|--------------------------|
| `GEMINI_API_KEY`      | Google AI Studio API key         | `AIzaSy…`                |
| `CHESS_LLM_MODEL`     | Model name                       | `gemini-2.5-flash`       |

Build a `src/llm_client.py` module exposing two functions:

```python
def get_chat_model(temperature: float = 0.1) -> BaseChatModel:
    """Return a LangChain chat model configured from env. Used for streaming."""

def call_llm(messages: list[dict], temperature: float = 0.2) -> str:
    """One-shot blocking completion. `messages` is OpenAI-style:
       [{'role': 'user'|'system'|'assistant', 'content': str}, ...]
    """
```

Architect this module so a second provider (LiteLLM, OpenAI direct, etc.) could be slotted in later via a `CHESS_LLM_PROVIDER` env var — but Phase 1 only implements Gemini. Phase 2 will add the multi-provider switch.

---

## File layout (Phase 1)

Create exactly these files. Nothing else.

```
personalgm-codex/
├── app.py                              # Streamlit entry point
├── requirements.txt                    # See "Tech stack" above
├── .env.example                        # Template; user fills in GEMINI_API_KEY
├── .gitignore                          # Sensible Python + Streamlit defaults
├── README.md                           # 1-page setup + run instructions
├── AGENTS.md                           # (this file — leave it alone)
└── src/
    ├── __init__.py                     # empty
    ├── state.py                        # LangGraph state schema (TypedDict)
    ├── supervisor.py                   # LangGraph supervisor + graph wiring
    ├── llm_client.py                   # Single LLM chokepoint
    ├── lichess_client.py               # Lichess HTTP + PGN parsing
    ├── board_render.py                 # FEN → SVG via python-chess
    ├── opening_expert.py               # Specialist #1 (streaming)
    └── opening_coach.py                # Specialist #2 (Repertoire Doctor)
```

Do **NOT** create `src/opponent_scout.py`, `src/postgame_analyst.py`, `src/chess_tools.py`, `src/cache.py`, or `src/practice_generator.py` — those are Phase 2.

---

## Architecture (Phase 1)

```
┌────────────────────────────────────────────────────────────────────┐
│  Streamlit UI  (app.py)                                            │
│  ┌──────────────┐    ┌────────────────────────────────────────┐    │
│  │   Sidebar    │    │   Main chat area                       │    │
│  │              │    │                                        │    │
│  │  ♟️ PersonalGM│    │   [Hero banner: "Your AI Chess Coach"]│    │
│  │              │    │                                        │    │
│  │  Features    │    │   Chat history (st.chat_message)       │    │
│  │  📊 Repertoire│ ──►│   ↳ User: "explain Caro-Kann"          │    │
│  │  ⚔️ Scout (P2)│    │   ↳ Assistant: <streamed lesson + SVGs>│    │
│  │  🔍 Analyse(P2)│   │                                        │    │
│  │              │    │   [Chat input box ────────────────────►]│    │
│  │  Settings    │    │                                        │    │
│  │  ☐ Dev mode  │    └────────────────────────────────────────┘    │
│  └──────────────┘                                                  │
└────────────────────────────────────────────────────────────────────┘
                           │
                           ▼
            ┌──────────────────────────────┐
            │  LangGraph supervisor        │
            │  (src/supervisor.py)         │
            │                              │
            │  intent ──► route to:        │
            │   ┌──ask_opening ──► Opening Expert  │
            │   ├──repertoire   ──► Opening Coach  │
            │   ├──scout        ──► (Phase 2 stub) │
            │   └──analyze_game ──► (Phase 2 stub) │
            └──────────────────────────────┘
                           │
                           ▼
            ┌──────────────────────────────┐
            │  LLM (Gemini 2.5 Flash)      │
            │  via src/llm_client.py       │
            └──────────────────────────────┘
```

---

## State schema — `src/state.py`

Use a `TypedDict`. LangGraph accumulates `trace_log` across nodes (use `Annotated[list[TraceEntry], operator.add]`); all other fields replace.

```python
class TraceEntry(TypedDict, total=False):
    ts: str           # ISO timestamp
    agent: str        # which specialist logged this
    tool: str         # tool name OR "node"
    args: dict        # short, JSON-serialisable
    result: str       # short summary
    status: str       # "ok" | "error" | "running"


class PersonalGMState(TypedDict, total=False):
    user_message: str                          # raw user input
    intent: str                                # one of VALID_INTENTS
    username: Optional[str]                    # Lichess username (for repertoire)
    current_opening: Optional[str]             # last opening discussed
    teach_mode: str                            # "general" | "prep_against_opponent"
    repertoire_data: Optional[dict]            # output of opening_coach
    opening_lesson: Optional[dict]             # output of opening_expert
    final_text: str                            # markdown the UI will render
    trace_log: Annotated[list[TraceEntry], operator.add]


VALID_INTENTS = (
    "ask_opening",       # Opening Expert
    "repertoire",        # Opening Coach
    "scout",             # Phase 2 (stub returns placeholder)
    "analyze_game",      # Phase 2 (stub returns placeholder)
)


def new_state(**overrides) -> PersonalGMState:
    """Factory with sane defaults."""
    base: PersonalGMState = {
        "user_message": "",
        "intent": "ask_opening",
        "teach_mode": "general",
        "trace_log": [],
        "final_text": "",
    }
    base.update(overrides)
    return base
```

---

## Supervisor — `src/supervisor.py`

Rule-based routing (no LLM classification needed in Phase 1 — the UI sets `intent` explicitly via the chat dispatch logic). Use `StateGraph(PersonalGMState)`.

**Nodes:**
- `supervisor_node` — normalises `state["intent"]`, logs a trace entry, returns `{"intent": <normalised>, "trace_log": [...]}`.
- `opening_expert_node` — calls `src.opening_expert.teach_opening(state["user_message"], state.get("teach_mode", "general"))`. Stores result in `state["opening_lesson"]` and renders to `state["final_text"]`.
- `opening_coach_node` — calls `src.opening_coach.analyze_repertoire(state["username"], n_games=50)`. Stores in `state["repertoire_data"]`, renders to `state["final_text"]`.
- `scout_stub_node` — returns `final_text` = `"⏸ **Opponent Scout** — coming in Phase 2. Not yet implemented."`
- `postgame_stub_node` — returns `final_text` = `"⏸ **Postgame Analyst** — coming in Phase 2. Not yet implemented."`

**Graph wiring:**

```
START → supervisor_node → (conditional edge by intent) → specialist_node → END
```

The conditional edge function `route_after_supervisor(state)` returns one of:
`"ask_opening" | "repertoire" | "scout" | "analyze_game"`.

Cache the compiled graph with `functools.lru_cache(maxsize=1)`:

```python
@lru_cache(maxsize=1)
def get_graph():
    graph = StateGraph(PersonalGMState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("ask_opening", opening_expert_node)
    graph.add_node("repertoire", opening_coach_node)
    graph.add_node("scout", scout_stub_node)
    graph.add_node("analyze_game", postgame_stub_node)
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", route_after_supervisor, {
        "ask_opening": "ask_opening",
        "repertoire": "repertoire",
        "scout": "scout",
        "analyze_game": "analyze_game",
    })
    for terminal in ("ask_opening", "repertoire", "scout", "analyze_game"):
        graph.add_edge(terminal, END)
    return graph.compile()
```

---

## Specialist #1 — Opening Expert  (`src/opening_expert.py`)

Given an opening name, ECO code, or move sequence, produces a structured 10-section lesson with deterministic SVG-rendered chess boards.

### Public API

```python
def teach_opening(opening_name: str, mode: str = "general") -> dict:
    """Blocking version. Returns {"markdown": str, "positions": [dict]}.
    
    `markdown` has {{position}} blocks that need rendering via render_lesson_html.
    `positions` is the parsed list of position dicts (move sequence, FEN, caption, SVG).
    """

def stream_opening(opening_name: str, mode: str = "general") -> Iterator[str]:
    """Streaming version. Yields markdown chunks as the LLM produces them.
    
    Used by the UI's st.write_stream(). The UI accumulates chunks, then
    after streaming completes, post-processes the full markdown to replace
    {{position}} blocks with rendered SVGs.
    """

def render_lesson_html(lesson: dict) -> str:
    """Take {markdown, positions} and return final HTML with {{position}} blocks
    replaced by centred SVG images + captions. Ready for st.markdown(..., unsafe_allow_html=True).
    """

def parse_position_blocks(markdown: str) -> list[dict]:
    """Extract {{position}}...{{/position}} blocks from raw LLM markdown.
    Returns list of dicts: {raw, moves, fen, caption}.
    """
```

### System prompt (USE VERBATIM — do not rewrite)

```
You are an Opening Expert — a chess master with encyclopedic knowledge of every
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
```

### `{{position}}` block parsing

Parse with this regex (DOTALL, IGNORECASE):
```python
POSITION_BLOCK_RE = re.compile(r"\{\{position\}\}(.*?)\{\{/position\}\}", re.DOTALL | re.IGNORECASE)
```

For each block, extract `moves`, `fen`, `caption` via per-key regex like:
```python
r"^\s*moves\s*:\s*(.+?)\s*$"   # MULTILINE | IGNORECASE
```

Strip wrapping single/double quotes from values (LLM sometimes adds them).

### FEN trust policy

LLMs hallucinate FENs. Apply this logic per position block:
1. If `chess.Board(fen)` parses without error → use the LLM's FEN.
2. Else → derive FEN by replaying `moves` via `chess.pgn.read_game()` / `Board().push_san()`. Use that FEN.
3. If both fail → skip rendering this position, leave a clean note in HTML.

Helpers live in `src/board_render.py` (see next section).

---

## Board rendering — `src/board_render.py`

```python
def fen_is_valid(fen: str) -> bool:
    """True iff chess.Board(fen) parses cleanly."""

def derive_fen_from_moves(moves_str: str) -> Optional[str]:
    """Replay '1.e4 e5 2.Nf3 ...' and return the final FEN. None on failure."""

def fen_with_fallback(fen: str, moves: str) -> Optional[str]:
    """Return fen if valid, else derive_fen_from_moves(moves), else None."""

def render_board_svg(
    fen: str,
    last_move: Optional[str] = None,
    arrows: Optional[Sequence] = None,
    size: int = 350,
) -> Optional[str]:
    """Return SVG markup string, or None if FEN unparseable. Uses chess.svg.board()."""
```

Use `chess.svg.board(board=chess.Board(fen), size=size, lastmove=lastmove_obj)`.

---

## Specialist #2 — Opening Coach / Repertoire Doctor  (`src/opening_coach.py`)

Pulls a user's last N Lichess games and identifies their weakest opening family.

### Pipeline

1. Call `src.lichess_client.fetch_user_games(username, n_games=50)`.
2. For each game, cluster the opening into a coarse family via `cluster_to_family(eco, opening_name)`. Lichess opening names follow `<Family>: <Variation>` — split on first colon to get family.
3. Tally per-family stats split by color played: `games, wins, draws, losses, score_pct, tier`.
   - `score_pct = (wins + 0.5 * draws) / games * 100`
   - Tiers (require min sample of 3 games to tier):
     - `score_pct >= 55` → `"Strong"`
     - `40 <= score_pct < 55` → `"Mixed"`
     - `score_pct < 40` → `"Weak"`
     - `games < 3` → `"Insufficient data"`
4. Recommend the WEAKEST family with the LARGEST sample (Weak tier only). `None` if no Weak family clears the 3-game bar.
5. Build a markdown report with per-color tables and the recommendation.

### Public API

```python
def cluster_to_family(eco: str, opening_name: str) -> str:
    """e.g. ('B12', 'Caro-Kann Defense: Advance Variation') → 'Caro-Kann Defense'.
    Treats '?', '-', '' as 'no opening info' → falls back to ECO code → 'Unclassified'."""

def analyze_repertoire(username: str, n_games: int = 50) -> dict:
    """Returns:
       {
         'username': str,
         'total_games': int,
         'by_color': {'white': [family_stat, ...], 'black': [family_stat, ...]},
         'recommended_opening': Optional[str],  # family name to study
         'recommended_color': Optional[str],    # 'white' | 'black'
       }
    where family_stat = {'family': str, 'games': int, 'wins': int, 'draws': int,
                         'losses': int, 'score_pct': float, 'tier': str}
    """

def render_repertoire_markdown(analysis: dict) -> str:
    """Markdown with tables per color + a recommendation footer.
    Use emoji tier markers: 🟢 Strong, 🟡 Mixed, 🔴 Weak, ⚪ Insufficient."""
```

---

## Lichess client — `src/lichess_client.py`

```python
LICHESS_BASE = "https://lichess.org"
PGN_ACCEPT = {"Accept": "application/x-chess-pgn"}
DEFAULT_TIMEOUT = 20


class LichessUserNotFoundError(Exception): ...


def fetch_user_games(username: str, n_games: int = 50) -> list[dict]:
    """GET /api/games/user/{username}?max={n_games}&pgnInJson=false&clocks=false&evals=false
    Accept: application/x-chess-pgn
    Parse the multi-game PGN stream. Return list of dicts with shape:
        {
          'pgn': str,
          'game_id': str,
          'eco': str,
          'opening_name': str,
          'result': '1-0' | '0-1' | '1/2-1/2' | '*',
          'color_played': 'white' | 'black' | 'unknown',
          'outcome': 'win' | 'loss' | 'draw' | 'unknown',
          'time_control': str,
          'date': str,
          'white': str,
          'black': str,
          'has_evals': bool,
        }
    """
```

`outcome` is from the perspective of the named user (case-insensitive match against `white`/`black`).

Use `chess.pgn.read_game(io.StringIO(pgn))` to iterate games.

`has_evals` is True if `[%eval` appears anywhere in the PGN body. Phase 1 doesn't use this; Phase 2 will.

Raise `LichessUserNotFoundError` on HTTP 404. Other errors → bubble up as `requests.HTTPError`.

---

## UI spec — `app.py`

### Page setup

```python
st.set_page_config(
    page_title="PersonalGM",
    layout="wide",
    page_icon="♟️",
    initial_sidebar_state="expanded",
)
```

Inject CSS for a sky-blue hero banner and rounded sidebar (Tailwind palette: `#0EA5E9` → `#38BDF8` gradient on the banner; `#E0F2FE` → `#F0F9FF` on the sidebar). Keep CSS minimal and clean — don't over-design.

### Sidebar

```
♟️ PersonalGM
─────────────
Features
  📊 Repertoire Doctor   [button → opens dialog: "Lichess username?"]
  ⚔️ Opponent Scout (Phase 2)   [disabled or shows "Coming soon"]
  🔍 Postgame Analyst (Phase 2)  [disabled or shows "Coming soon"]
─────────────
Settings
  ☐ Dev mode (toggle)
  🧹 Clear chat (button)
─────────────
LLM: Gemini · gemini-2.5-flash
```

The Repertoire Doctor button opens an `st.dialog` modal that asks for the Lichess username (text input) + an n_games slider (10–200, default 50). On submit, dispatch the request through the LangGraph supervisor with `intent="repertoire"`.

Scout / Postgame buttons can either be visually disabled or show a short "Phase 2" note when clicked.

### Main area

```
[Hero banner: "♟️ PersonalGM — Your AI Chess Coach"]

[Dev mode tool-trace expander — only visible if dev mode toggled ON]

(Chat history — st.chat_message per turn)
  User:      explain the Caro-Kann Advance
  Assistant: <streamed markdown lesson with rendered SVG boards>
  
[Chat input box: "Ask anything..."]
```

### Chat dispatch logic

When the user submits text via `st.chat_input`:

1. Append to `st.session_state.messages` as `{"role": "user", "content": text}`.
2. Run a tiny **intent classifier** — pure regex/keyword, no LLM:
   - If text contains words like "explain", "teach me about", "what is", "show me", an ECO code (regex `[ABCDE]\d{2}`), or a known opening name → `intent = "ask_opening"`.
   - Otherwise default to `"ask_opening"` (the most general specialist).
3. Build state: `new_state(user_message=text, intent=intent)`.
4. Get graph: `get_graph()` from `src.supervisor`.
5. Stream the response:
   - For `intent == "ask_opening"`: use `stream_opening(text)` directly (not via the graph) because LangGraph's non-async streaming is awkward in Streamlit. Wrap with `st.write_stream()`. After streaming completes, call `render_lesson_html(...)` on the full text and replace the placeholder with the final rendered HTML (so SVG boards appear).
   - For `intent == "repertoire"`: this comes via the sidebar dialog, not the chat input. Run the graph blocking (`graph.invoke(state)`), display the resulting markdown.
6. Append assistant response to `st.session_state.messages`.

### Dev mode

When toggled on, show a collapsible "🛠 Trace" expander between the hero and chat. Render the `trace_log` from the last state as a small table. End users never see this; it's for hackathon demo when you want to show the supervisor → specialist routing live.

---

## Environment & secrets — `.env.example`

```dotenv
# LLM provider — Phase 1 supports gemini only
CHESS_LLM_PROVIDER=gemini
CHESS_LLM_MODEL=gemini-2.5-flash

# Get a free key at https://aistudio.google.com/apikey
GEMINI_API_KEY=your-gemini-api-key-here
```

In `.gitignore` include `.env`, `.venv/`, `__pycache__/`, `.streamlit/secrets.toml`, plus the standard Python ignores.

---

## `README.md` (1 page max)

Should include:
- One-paragraph intro
- Prerequisites (Python 3.12)
- Setup steps: clone, venv, `pip install -r requirements.txt`, copy `.env.example` to `.env` and fill in Gemini key
- Run: `streamlit run app.py`
- Phase 1 status: Opening Expert ✅, Repertoire Doctor ✅, Opponent Scout ⏸, Postgame Analyst ⏸
- Phase 2 preview (one sentence): "Phase 2 adds Opponent Scout, Postgame Analyst with Stockfish, disk-backed caching for instant demo responses, and full streaming across all specialists."

---

## Conventions

- **Style:** PEP 8, type hints, docstrings on public functions.
- **Imports:** Module-level imports for cheap stuff; lazy-import heavy stuff (LangGraph compile, LLM model construction) inside the first function call that needs it, so Streamlit's first paint isn't blocked.
- **Error handling:** User-facing errors (bad Lichess username, missing API key) should render as a friendly red Streamlit error. Stack traces only when dev mode is ON.
- **No global mutable state** except `st.session_state`.
- **No prints in library code.** Use `st.write` or `st.error` for UI; bubble exceptions upward otherwise.

---

## Phase 1 — Acceptance checklist

You're done with Phase 1 when ALL these pass:

- [ ] `streamlit run app.py` starts without errors
- [ ] Hero banner + sidebar render correctly
- [ ] Typing **`explain the Caro-Kann Advance`** in the chat streams a 10-section lesson with at least one SVG board image
- [ ] Sidebar **Repertoire Doctor** button opens a dialog; submitting username **`DrNykterstein`** (or any valid Lichess user) returns a per-color opening breakdown table + a recommended opening to study
- [ ] Sidebar **Opponent Scout** and **Postgame Analyst** buttons show a clean "Coming in Phase 2" message — they do NOT 500 or crash
- [ ] Dev mode toggle shows a tool-trace expander; with it on, you can see the supervisor → specialist routing for each turn
- [ ] No Stockfish dependency anywhere
- [ ] No `cache.py` file (cache is Phase 2)
- [ ] All files listed in the file layout exist; no extras
- [ ] `requirements.txt` is exactly the libraries from "Tech stack" — no more, no less
- [ ] `git status` would show only source code (no `.env`, no `.venv/`, no `__pycache__`)

---

## 🛑 STOP RULE — Critical

After Phase 1 is complete and all acceptance items pass:

1. Run `streamlit run app.py` and verify the smoke tests above.
2. Print a clear `✅ Phase 1 complete. Awaiting "go" from user before starting Phase 2.` message.
3. **STOP. Do not start Phase 2.**

The user will inspect the result manually, run the app themselves, and explicitly say **"go ahead with Phase 2"** before you proceed. Until that signal, do not create new files, do not modify any existing files. You may answer questions about the code, but no edits.

---

## Phase 2 — Build LATER (after user gives the go-ahead)

Do not read this section until Phase 1 is signed off and the user says "go".

### Phase 2 deliverables

1. **`src/cache.py`** — Disk-backed pickle cache with TTL (7 days default) and version invalidation (`CACHE_VERSION = "v1"`). Exposes `cache_get`, `cache_set`, and a `@cached(namespace, key_fn, ttl_days)` decorator. CLI for `python -m src.cache stats` and `python -m src.cache clear [namespace]`.

2. **Wrap existing specialists with `@cached`:**
   - `teach_opening(opening_name, mode)` → namespace `"openings"`, key = `f"{mode}__{normalize(opening_name)}"`
   - `analyze_repertoire(username, n_games)` → namespace `"repertoire"`, key = `f"{username.lower()}__{n_games}"`

3. **UI cache short-circuit** — When the user submits a chat that matches a cached opening, render with an "⚡ cached" badge and skip the LLM call. For Repertoire Doctor, same idea via cached results.

4. **`src/chess_tools.py`** — Stockfish wrapper. Reads `STOCKFISH_PATH` env var. Functions:
   - `analyze_position(fen, depth=15, time_limit_s=2.0) -> dict` (eval, best move, PV)
   - `score_move(fen, move_san, depth=15) -> dict` (eval before, eval after, classification: brilliant / good / inaccuracy / mistake / blunder)
   - Use `python-chess`'s `chess.engine.SimpleEngine.popen_uci()`.

5. **`src/opponent_scout.py`** — Specialist #3. Pulls opponent's last 100 games via `fetch_user_games`, finds their most-frequently-played openings (top 3 as White, top 3 as Black), tags strong/weak openings using the same tier rules as the Repertoire Doctor. Returns a markdown briefing + recommended preparation.

6. **`src/postgame_analyst.py`** — Specialist #4. Takes a Lichess game URL or raw PGN. Fetches/parses PGN, runs Stockfish analysis on every move (depth 15, ~2s per move), flags mistakes/blunders (eval drops >100cp), generates LLM commentary on the top 3 critical moments. Returns markdown + per-move table + an embedded "game report visual" (multi-board with annotations).

7. **Wire all 4 specialists into the supervisor.** Remove the stub nodes for scout/analyze_game; replace with real implementations.

8. **Streaming for Repertoire Doctor and Postgame Analyst** where it makes UX sense (Postgame yes, Repertoire less so since it's mostly tabular).

9. **Add `litellm` and `langchain-openai` to requirements.txt** + extend `llm_client.py` to support `CHESS_LLM_PROVIDER=litellm` (corporate proxy via OpenAI-compatible base URL). Keeps Gemini as the default for personal-laptop dev; LiteLLM is for corporate.

10. **Deployment files** — `Dockerfile` and `packages.txt` for HuggingFace Spaces. Stockfish installs via `apt-get` (binary at `/usr/games/stockfish`).

### Phase 2 acceptance checklist (not exhaustive)

- All 4 specialists work end-to-end
- Cache hits render in <100 ms with "⚡ cached" badge
- Stockfish analysis completes in <30 s for a 40-move game
- Lichess URL paste auto-triggers Postgame Analyst
- "scout {username}" routes to Opponent Scout via supervisor
- A second LLM provider (LiteLLM) can be selected via env without code changes

---

## Final reminder

You're building Phase 1 only. Read the acceptance checklist before starting, then build top-down: create the file layout, fill `state.py` first (so types are correct everywhere), then `llm_client.py`, then `lichess_client.py` + `board_render.py` (pure utilities), then `opening_expert.py` and `opening_coach.py`, then `supervisor.py`, then `app.py` last (it depends on everything else).

When in doubt, ask the user — don't invent. Especially: don't add libraries, don't add files outside the layout, don't touch Phase 2 scope.

🛑 **STOP after Phase 1. Wait for "go ahead" before Phase 2.**
