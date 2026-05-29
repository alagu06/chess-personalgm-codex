"""Postgame Analyst specialist."""

from __future__ import annotations

import io
import os
import re

import chess
import chess.pgn
import requests

from src.chess_tools import StockfishUnavailableError, classify_eval_drop
from src.llm_client import call_llm

LICHESS_GAME_RE = re.compile(r"lichess\.org/(?:game/export/)?([A-Za-z0-9]{8,12})")


def _pgn_from_input(game_input: str) -> str:
    text = game_input.strip()
    match = LICHESS_GAME_RE.search(text)
    if not match:
        return text
    game_id = match.group(1)
    response = requests.get(
        f"https://lichess.org/game/export/{game_id}",
        params={"clocks": "false", "evals": "false"},
        headers={"Accept": "application/x-chess-pgn"},
        timeout=20,
    )
    response.raise_for_status()
    return response.text


def _parse_game(pgn: str) -> chess.pgn.Game:
    game = chess.pgn.read_game(io.StringIO(pgn))
    if game is None:
        raise ValueError("Could not parse PGN or Lichess game URL.")
    return game


def _analyze_game_moves(
    game: chess.pgn.Game,
    depth: int,
    time_limit_s: float,
) -> list[dict]:
    from src.chess_tools import _analyze_with_engine, _engine

    rows = []
    board = game.board()
    with _engine() as engine:
        for ply, move in enumerate(game.mainline_moves(), start=1):
            san = board.san(move)
            mover = board.turn
            before = _analyze_with_engine(engine, board, depth, time_limit_s)
            board.push(move)
            after = _analyze_with_engine(engine, board, depth, time_limit_s)
            before_for_mover = before["eval_cp"] if mover == chess.WHITE else -before["eval_cp"]
            after_for_mover = after["eval_cp"] if mover == chess.WHITE else -after["eval_cp"]
            drop = before_for_mover - after_for_mover
            rows.append(
                {
                    "ply": ply,
                    "move": san,
                    "side": "White" if mover == chess.WHITE else "Black",
                    "eval_before_cp": before["eval_cp"],
                    "eval_after_cp": after["eval_cp"],
                    "drop_cp": drop,
                    "classification": classify_eval_drop(drop),
                    "best_move": before["best_move"],
                    "fen_after": board.fen(),
                }
            )
    return rows


def _critical_moments(rows: list[dict], limit: int = 3) -> list[dict]:
    bad = [row for row in rows if row["classification"] in {"mistake", "blunder"}]
    return sorted(bad, key=lambda row: row["drop_cp"], reverse=True)[:limit]


def _llm_commentary(headers: chess.pgn.Headers, moments: list[dict]) -> str:
    if not moments:
        return "No major tactical swings were detected by Stockfish."
    summary = "\n".join(
        f"Ply {m['ply']} {m['side']} {m['move']}: {m['classification']}, "
        f"drop {m['drop_cp']}cp, best was {m['best_move']}"
        for m in moments
    )
    try:
        return call_llm(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a concise chess coach. Explain the listed critical "
                        "moments in practical club-player language."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Game: {headers.get('White')} vs {headers.get('Black')}\n{summary}",
                },
            ],
            temperature=0.2,
        )
    except Exception:
        return summary


def _table(rows: list[dict]) -> str:
    lines = [
        "| Ply | Side | Move | Eval before | Eval after | Loss | Class | Best |",
        "|---:|---|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {ply} | {side} | {move} | {before} | {after} | {drop} | {label} | {best} |".format(
                ply=row["ply"],
                side=row["side"],
                move=row["move"],
                before=row["eval_before_cp"],
                after=row["eval_after_cp"],
                drop=row["drop_cp"],
                label=row["classification"],
                best=row["best_move"],
            )
        )
    return "\n".join(lines)


def analyze_game(game_input: str, depth: int = 15, time_limit_s: float | None = None) -> dict:
    """Analyze a Lichess URL or raw PGN with Stockfish and LLM commentary."""

    if time_limit_s is None:
        time_limit_s = float(os.getenv("CHESS_STOCKFISH_TIME_LIMIT", "0.25"))
    pgn = _pgn_from_input(game_input)
    game = _parse_game(pgn)
    try:
        rows = _analyze_game_moves(game, depth=depth, time_limit_s=time_limit_s)
    except StockfishUnavailableError:
        raise
    moments = _critical_moments(rows)
    commentary = _llm_commentary(game.headers, moments)
    return {
        "headers": dict(game.headers),
        "rows": rows,
        "critical_moments": moments,
        "commentary": commentary,
    }


def render_postgame_markdown(analysis: dict) -> str:
    """Render postgame analysis as markdown."""

    headers = analysis["headers"]
    white = headers.get("White", "White")
    black = headers.get("Black", "Black")
    event = headers.get("Event", "Game")
    moments = analysis["critical_moments"]
    rows = moments if moments else analysis["rows"][:10]
    return "\n\n".join(
        [
            f"## Postgame Analyst: {white} vs {black}",
            f"**Event:** {event}  \n**Result:** {headers.get('Result', '*')}",
            "### Coach Commentary",
            analysis["commentary"],
            "### Critical Moments",
            _table(rows),
        ]
    )
