"""Chess board parsing and SVG rendering helpers."""

from __future__ import annotations

import io
from typing import Optional, Sequence

import chess
import chess.pgn
import chess.svg


def fen_is_valid(fen: str) -> bool:
    """True iff chess.Board(fen) parses cleanly."""

    if not fen:
        return False
    try:
        chess.Board(fen)
    except ValueError:
        return False
    return True


def derive_fen_from_moves(moves_str: str) -> Optional[str]:
    """Replay SAN moves and return the final FEN. None on failure."""

    if not moves_str:
        return None

    pgn_text = f'[Event "?"]\n\n{moves_str} *'
    try:
        game = chess.pgn.read_game(io.StringIO(pgn_text))
        if game is not None:
            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
            return board.fen()
    except Exception:
        pass

    board = chess.Board()
    try:
        for token in moves_str.replace("\n", " ").split():
            clean = token.strip()
            if not clean or clean.endswith(".") or clean in {"*", "1-0", "0-1", "1/2-1/2"}:
                continue
            if "." in clean:
                clean = clean.split(".")[-1]
            if not clean:
                continue
            board.push_san(clean)
        return board.fen()
    except Exception:
        return None


def fen_with_fallback(fen: str, moves: str) -> Optional[str]:
    """Return fen if valid, else derive_fen_from_moves(moves), else None."""

    if fen_is_valid(fen):
        return fen
    return derive_fen_from_moves(moves)


def render_board_svg(
    fen: str,
    last_move: Optional[str] = None,
    arrows: Optional[Sequence] = None,
    size: int = 350,
) -> Optional[str]:
    """Return SVG markup string, or None if FEN is unparseable."""

    try:
        board = chess.Board(fen)
    except ValueError:
        return None

    lastmove_obj = None
    if last_move:
        try:
            lastmove_obj = board.parse_san(last_move)
        except ValueError:
            try:
                lastmove_obj = chess.Move.from_uci(last_move)
            except ValueError:
                lastmove_obj = None

    return chess.svg.board(
        board=board,
        size=size,
        lastmove=lastmove_obj,
        arrows=arrows,
    )
