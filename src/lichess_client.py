"""Small Lichess PGN client used by the Repertoire Doctor."""

from __future__ import annotations

import io
from typing import Iterator

import chess.pgn
import requests

LICHESS_BASE = "https://lichess.org"
PGN_ACCEPT = {"Accept": "application/x-chess-pgn"}
DEFAULT_TIMEOUT = 20


class LichessUserNotFoundError(Exception):
    """Raised when Lichess returns 404 for a username."""


def _split_pgn_games(pgn_text: str) -> Iterator[str]:
    current: list[str] = []
    for line in pgn_text.splitlines():
        if line.startswith("[Event ") and current:
            yield "\n".join(current).strip()
            current = []
        current.append(line)
    if current:
        yield "\n".join(current).strip()


def _headers_from_pgn(pgn: str) -> chess.pgn.Headers:
    game = chess.pgn.read_game(io.StringIO(pgn))
    if game is None:
        return chess.pgn.Headers()
    return game.headers


def _outcome_for_user(result: str, color_played: str) -> str:
    if result == "1/2-1/2":
        return "draw"
    if color_played == "white":
        return "win" if result == "1-0" else "loss" if result == "0-1" else "unknown"
    if color_played == "black":
        return "win" if result == "0-1" else "loss" if result == "1-0" else "unknown"
    return "unknown"


def fetch_user_games(username: str, n_games: int = 50) -> list[dict]:
    """Fetch and parse a user's recent games from Lichess."""

    clean_username = username.strip()
    url = f"{LICHESS_BASE}/api/games/user/{clean_username}"
    params = {
        "max": n_games,
        "pgnInJson": "false",
        "clocks": "false",
        "evals": "false",
        "opening": "true",
    }
    response = requests.get(
        url,
        params=params,
        headers=PGN_ACCEPT,
        timeout=DEFAULT_TIMEOUT,
    )
    if response.status_code == 404:
        raise LichessUserNotFoundError(f"Lichess user not found: {clean_username}")
    response.raise_for_status()

    parsed_games = []
    for pgn in _split_pgn_games(response.text):
        headers = _headers_from_pgn(pgn)
        white = headers.get("White", "")
        black = headers.get("Black", "")
        lower_name = clean_username.lower()
        if white.lower() == lower_name:
            color_played = "white"
        elif black.lower() == lower_name:
            color_played = "black"
        else:
            color_played = "unknown"

        result = headers.get("Result", "*")
        site = headers.get("Site", "")
        game_id = site.rstrip("/").split("/")[-1] if site else headers.get("LichessID", "")

        parsed_games.append(
            {
                "pgn": pgn,
                "game_id": game_id,
                "eco": headers.get("ECO", ""),
                "opening_name": headers.get("Opening", ""),
                "result": result,
                "color_played": color_played,
                "outcome": _outcome_for_user(result, color_played),
                "time_control": headers.get("TimeControl", ""),
                "date": headers.get("UTCDate", headers.get("Date", "")),
                "white": white,
                "black": black,
                "has_evals": "[%eval" in pgn,
            }
        )
    return parsed_games
