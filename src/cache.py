"""Disk-backed cache for PersonalGM specialists."""

from __future__ import annotations

import argparse
import functools
import hashlib
import os
import pickle
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

CACHE_VERSION = "v1"
CACHE_ROOT = Path(os.getenv("PERSONALGM_CACHE_DIR", ".personalgm_cache"))


def normalize_key(value: str) -> str:
    """Normalize a user-facing value into a stable cache key fragment."""

    return "_".join((value or "").strip().lower().split())


def _namespace_dir(namespace: str) -> Path:
    return CACHE_ROOT / CACHE_VERSION / namespace


def _cache_path(namespace: str, key: str) -> Path:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return _namespace_dir(namespace) / f"{digest}.pkl"


def cache_get(namespace: str, key: str, ttl_days: int = 7) -> Any:
    """Return cached value or None when absent, expired, or invalid."""

    path = _cache_path(namespace, key)
    if not path.exists():
        return None
    max_age_s = ttl_days * 24 * 60 * 60
    if time.time() - path.stat().st_mtime > max_age_s:
        return None
    try:
        with path.open("rb") as handle:
            return pickle.load(handle)
    except (OSError, pickle.PickleError, EOFError):
        return None


def cache_set(namespace: str, key: str, value: Any) -> None:
    """Persist value in the cache."""

    path = _cache_path(namespace, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    with tmp_path.open("wb") as handle:
        pickle.dump(value, handle)
    tmp_path.replace(path)


def cached(namespace: str, key_fn: Callable[..., str], ttl_days: int = 7):
    """Cache decorator for pure-ish specialist functions."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = key_fn(*args, **kwargs)
            cached_value = cache_get(namespace, key, ttl_days=ttl_days)
            if cached_value is not None:
                if isinstance(cached_value, dict):
                    cached_value = dict(cached_value)
                    cached_value["_cached"] = True
                return cached_value
            value = func(*args, **kwargs)
            cache_set(namespace, key, value)
            if isinstance(value, dict):
                value = dict(value)
                value["_cached"] = False
            return value

        return wrapper

    return decorator


def _stats() -> str:
    if not CACHE_ROOT.exists():
        return "Cache is empty."
    lines = []
    for namespace_dir in sorted((CACHE_ROOT / CACHE_VERSION).glob("*")):
        if namespace_dir.is_dir():
            count = len(list(namespace_dir.glob("*.pkl")))
            lines.append(f"{namespace_dir.name}: {count} entries")
    return "\n".join(lines) if lines else "Cache is empty."


def _clear(namespace: str | None) -> str:
    target = _namespace_dir(namespace) if namespace else CACHE_ROOT / CACHE_VERSION
    if target.exists():
        shutil.rmtree(target)
    return f"Cleared {namespace or 'all'} cache."


def main() -> None:
    """Small CLI for cache stats and clearing."""

    parser = argparse.ArgumentParser(description="PersonalGM cache utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("stats")
    clear_parser = subparsers.add_parser("clear")
    clear_parser.add_argument("namespace", nargs="?")
    args = parser.parse_args()

    if args.command == "stats":
        print(_stats())
    elif args.command == "clear":
        print(_clear(args.namespace))


if __name__ == "__main__":
    main()
