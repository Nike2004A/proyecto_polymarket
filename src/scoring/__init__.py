"""API pública ligera para scoring."""

__all__ = [
    "score_active_markets",
    "generate_signals",
    "score_active_markets_ts",
    "score_resolved_markets_ts",
]


def __getattr__(name: str):
    if name == "score_active_markets":
        from .scorer import score_active_markets

        return score_active_markets
    if name == "generate_signals":
        from .signals import generate_signals

        return generate_signals
    if name in {"score_active_markets_ts", "score_resolved_markets_ts"}:
        from .ts_scorer import score_active_markets_ts, score_resolved_markets_ts

        return {
            "score_active_markets_ts": score_active_markets_ts,
            "score_resolved_markets_ts": score_resolved_markets_ts,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
