# PersonalGM

PersonalGM is a Streamlit web app for amateur chess improvement. It includes an AI Opening Expert, Repertoire Doctor, Opponent Scout, and Postgame Analyst orchestrated through a LangGraph supervisor.

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

For Postgame Analyst, install Stockfish and set `STOCKFISH_PATH` in `.env`. For a LiteLLM/OpenAI-compatible proxy, set `CHESS_LLM_PROVIDER=litellm`, `CHESS_LLM_MODEL`, `OPENAI_API_KEY`, and `OPENAI_API_BASE`.

## Run

```powershell
streamlit run app.py
```

## Status

- Opening Expert: ✅
- Repertoire Doctor: ✅
- Opponent Scout: ✅
- Postgame Analyst: ✅
- Disk cache: ✅
- Gemini and LiteLLM provider selection: ✅

Cache utilities:

```powershell
python -m src.cache stats
python -m src.cache clear
python -m src.cache clear openings
```
