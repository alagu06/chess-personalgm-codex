# PersonalGM

PersonalGM is a Streamlit web app for amateur chess improvement. Phase 1 includes an AI Opening Expert for structured opening lessons with rendered chess boards, plus a Repertoire Doctor that reviews recent Lichess games and identifies weak opening families.

## Prerequisites

- Python 3.12

## Setup

```powershell
git clone <repo-url>
cd personalgm-codex
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and set `GEMINI_API_KEY` to your Google AI Studio key.

## Run

```powershell
streamlit run app.py
```

## Phase 1 Status

- Opening Expert: ✅
- Repertoire Doctor: ✅
- Opponent Scout: ⏸
- Postgame Analyst: ⏸

Phase 2 adds Opponent Scout, Postgame Analyst with Stockfish, disk-backed caching for instant demo responses, and full streaming across all specialists.
