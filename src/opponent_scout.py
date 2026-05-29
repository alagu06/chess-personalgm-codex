"""Opponent Scout specialist."""

from __future__ import annotations

from src.lichess_client import fetch_user_games
from src.opening_coach import TIER_MARKERS, _summarize_bucket, cluster_to_family


def scout_opponent(username: str, n_games: int = 100) -> dict:
    """Analyze an opponent's recent opening choices."""

    games = fetch_user_games(username, n_games=n_games)
    buckets = {
        "white": {},
        "black": {},
    }
    for game in games:
        color = game.get("color_played")
        if color not in buckets:
            continue
        family = cluster_to_family(game.get("eco", ""), game.get("opening_name", ""))
        row = buckets[color].setdefault(
            family,
            {"games": 0, "wins": 0, "draws": 0, "losses": 0},
        )
        row["games"] += 1
        outcome = game.get("outcome")
        if outcome == "win":
            row["wins"] += 1
        elif outcome == "draw":
            row["draws"] += 1
        elif outcome == "loss":
            row["losses"] += 1

    by_color = {
        "white": _summarize_bucket(buckets["white"])[:3],
        "black": _summarize_bucket(buckets["black"])[:3],
    }
    return {
        "username": username,
        "total_games": len(games),
        "by_color": by_color,
    }


def _table(rows: list[dict]) -> str:
    if not rows:
        return "_No games found._"
    lines = [
        "| Opening family | Games | Score | Tier |",
        "|---|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['family']} | {row['games']} | {row['score_pct']:.1f}% | "
            f"{TIER_MARKERS[row['tier']]} |"
        )
    return "\n".join(lines)


def render_scout_markdown(analysis: dict) -> str:
    """Render a concise opponent preparation briefing."""

    username = analysis["username"]
    white = analysis["by_color"]["white"]
    black = analysis["by_color"]["black"]
    likely_white = white[0]["family"] if white else "unknown"
    likely_black = black[0]["family"] if black else "unknown"
    return "\n\n".join(
        [
            f"## Opponent Scout: {username}",
            f"Analyzed **{analysis['total_games']}** recent Lichess games.",
            "### When They Have White",
            _table(white),
            "### When They Have Black",
            _table(black),
            "### Preparation",
            (
                f"Expect **{likely_white}** when they play White and **{likely_black}** "
                "when they play Black. Prepare one compact response line for each, "
                "then review the lowest-scoring family above for practical chances."
            ),
        ]
    )
