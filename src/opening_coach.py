"""Repertoire Doctor specialist."""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

from src.lichess_client import fetch_user_games

TIER_MARKERS = {
    "Strong": "🟢 Strong",
    "Mixed": "🟡 Mixed",
    "Weak": "🔴 Weak",
    "Insufficient data": "⚪ Insufficient",
}


def cluster_to_family(eco: str, opening_name: str) -> str:
    """Collapse a Lichess opening name into a coarse family."""

    clean_opening = (opening_name or "").strip()
    if clean_opening and clean_opening not in {"?", "-"}:
        return clean_opening.split(":", 1)[0].strip() or "Unclassified"

    clean_eco = (eco or "").strip()
    if clean_eco and clean_eco not in {"?", "-"}:
        return clean_eco
    return "Unclassified"


def _tier(games: int, score_pct: float) -> str:
    if games < 3:
        return "Insufficient data"
    if score_pct >= 55:
        return "Strong"
    if score_pct >= 40:
        return "Mixed"
    return "Weak"


def _summarize_bucket(bucket: dict[str, dict]) -> list[dict]:
    stats = []
    for family, row in bucket.items():
        games = row["games"]
        score_pct = ((row["wins"] + 0.5 * row["draws"]) / games * 100) if games else 0
        stats.append(
            {
                "family": family,
                "games": games,
                "wins": row["wins"],
                "draws": row["draws"],
                "losses": row["losses"],
                "score_pct": round(score_pct, 1),
                "tier": _tier(games, score_pct),
            }
        )
    return sorted(stats, key=lambda item: (-item["games"], item["score_pct"], item["family"]))


def analyze_repertoire(username: str, n_games: int = 50) -> dict:
    """Analyze a Lichess user's opening results by color."""

    games = fetch_user_games(username, n_games=n_games)
    buckets: dict[str, dict[str, dict]] = {
        "white": defaultdict(lambda: {"games": 0, "wins": 0, "draws": 0, "losses": 0}),
        "black": defaultdict(lambda: {"games": 0, "wins": 0, "draws": 0, "losses": 0}),
    }

    for game in games:
        color = game.get("color_played", "unknown")
        if color not in buckets:
            continue
        family = cluster_to_family(game.get("eco", ""), game.get("opening_name", ""))
        row = buckets[color][family]
        row["games"] += 1
        outcome = game.get("outcome")
        if outcome == "win":
            row["wins"] += 1
        elif outcome == "draw":
            row["draws"] += 1
        elif outcome == "loss":
            row["losses"] += 1

    by_color = {
        "white": _summarize_bucket(buckets["white"]),
        "black": _summarize_bucket(buckets["black"]),
    }

    recommendation = _find_recommendation(by_color)
    return {
        "username": username,
        "total_games": len(games),
        "by_color": by_color,
        "recommended_opening": recommendation["family"] if recommendation else None,
        "recommended_color": recommendation["color"] if recommendation else None,
    }


def _find_recommendation(by_color: dict[str, list[dict]]) -> Optional[dict]:
    weak = []
    for color, rows in by_color.items():
        for row in rows:
            if row["tier"] == "Weak" and row["games"] >= 3:
                weak.append({**row, "color": color})
    if not weak:
        return None
    return sorted(weak, key=lambda row: (-row["games"], row["score_pct"], row["family"]))[0]


def _table(rows: list[dict]) -> str:
    if not rows:
        return "_No games found for this color._"

    lines = [
        "| Opening family | Games | W | D | L | Score | Tier |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {family} | {games} | {wins} | {draws} | {losses} | {score_pct:.1f}% | {tier} |".format(
                family=row["family"],
                games=row["games"],
                wins=row["wins"],
                draws=row["draws"],
                losses=row["losses"],
                score_pct=row["score_pct"],
                tier=TIER_MARKERS[row["tier"]],
            )
        )
    return "\n".join(lines)


def render_repertoire_markdown(analysis: dict) -> str:
    """Render repertoire analysis as markdown tables."""

    username = analysis.get("username", "")
    total_games = analysis.get("total_games", 0)
    by_color = analysis.get("by_color", {"white": [], "black": []})
    recommended_opening = analysis.get("recommended_opening")
    recommended_color = analysis.get("recommended_color")

    parts = [
        f"## Repertoire Doctor: {username}",
        f"Analyzed **{total_games}** recent Lichess games.",
        "### As White",
        _table(by_color.get("white", [])),
        "### As Black",
        _table(by_color.get("black", [])),
    ]

    if recommended_opening and recommended_color:
        parts.append(
            "### Recommendation\n"
            f"Study **{recommended_opening}** as **{recommended_color}** first. "
            "It has the weakest score among openings with enough games to trust the sample."
        )
    else:
        parts.append(
            "### Recommendation\n"
            "No weak opening family had at least 3 games, so there is no reliable single target yet."
        )

    return "\n\n".join(parts)
