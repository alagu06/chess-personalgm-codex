"""Stockfish-backed chess analysis helpers."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

import chess
import chess.engine


class StockfishUnavailableError(RuntimeError):
    """Raised when Stockfish cannot be launched."""


def _stockfish_path() -> str:
    path = os.getenv("STOCKFISH_PATH") or "/usr/games/stockfish"
    if os.name == "nt" and not os.getenv("STOCKFISH_PATH"):
        raise StockfishUnavailableError(
            "Set STOCKFISH_PATH in .env to your stockfish.exe path."
        )
    return path


@contextmanager
def _engine() -> Iterator[chess.engine.SimpleEngine]:
    path = _stockfish_path()
    try:
        engine = chess.engine.SimpleEngine.popen_uci(path)
    except Exception as exc:
        raise StockfishUnavailableError(f"Could not start Stockfish at {path}.") from exc
    try:
        yield engine
    finally:
        engine.quit()


def _score_to_cp(score: chess.engine.PovScore) -> int:
    white_score = score.white()
    mate = white_score.mate()
    if mate is not None:
        return 100000 if mate > 0 else -100000
    cp = white_score.score()
    return int(cp or 0)


def _analyze_with_engine(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    depth: int,
    time_limit_s: float,
) -> dict:
    limit = chess.engine.Limit(depth=depth, time=time_limit_s)
    info = engine.analyse(board, limit)
    pv = info.get("pv", [])
    best_move = pv[0] if pv else None
    pv_board = board.copy()
    pv_san = []
    for move in pv[:6]:
        pv_san.append(pv_board.san(move))
        pv_board.push(move)
    return {
        "fen": board.fen(),
        "eval_cp": _score_to_cp(info["score"]),
        "best_move": board.san(best_move) if best_move else "",
        "pv": " ".join(pv_san),
    }


def analyze_position(fen: str, depth: int = 15, time_limit_s: float = 2.0) -> dict:
    """Analyze a FEN and return eval, best move, and principal variation."""

    board = chess.Board(fen)
    with _engine() as engine:
        return _analyze_with_engine(engine, board, depth, time_limit_s)


def classify_eval_drop(drop_cp: int) -> str:
    """Classify a centipawn loss from the mover's perspective."""

    if drop_cp <= -150:
        return "brilliant"
    if drop_cp < 50:
        return "good"
    if drop_cp < 100:
        return "inaccuracy"
    if drop_cp < 300:
        return "mistake"
    return "blunder"


def score_move(fen: str, move_san: str, depth: int = 15, time_limit_s: float = 2.0) -> dict:
    """Score one SAN move from a FEN."""

    board = chess.Board(fen)
    mover = board.turn
    move = board.parse_san(move_san)
    with _engine() as engine:
        before = _analyze_with_engine(engine, board, depth, time_limit_s)
        board.push(move)
        after = _analyze_with_engine(engine, board, depth, time_limit_s)

    before_cp = before["eval_cp"] if mover == chess.WHITE else -before["eval_cp"]
    after_cp = after["eval_cp"] if mover == chess.WHITE else -after["eval_cp"]
    drop = before_cp - after_cp
    return {
        "move": move_san,
        "eval_before_cp": before["eval_cp"],
        "eval_after_cp": after["eval_cp"],
        "drop_cp": drop,
        "classification": classify_eval_drop(drop),
        "best_move": before["best_move"],
    }
