"""Signal generation from calibrated p_yes and EV."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_signals(
    scored_df: pd.DataFrame,
    buy_ev_threshold: float = 0.03,
    strong_buy_ev_threshold: float = 0.07,
    buy_roi_threshold: float = 0.10,
    strong_buy_roi_threshold: float = 0.20,
    expected_roi_cap: float = 5.0,
    min_price_yes: float = 0.03,
    max_price_yes: float = 0.85,
    max_days_to_end: float = 180.0,
    min_liquidity: float = 1000.0,
    min_volume_24h: float = 100.0,
    max_spread: float = 0.10,
    buy_threshold: float | None = None,
    strong_buy_threshold: float | None = None,
) -> pd.DataFrame:
    """Generate BUY/HOLD labels using calibrated EV thresholds."""
    df = scored_df.copy()
    df["signal"] = "HOLD"

    # Backward-compatible legacy mode for old notebooks that still pass raw `model_score`.
    if "ev_per_share" not in df.columns or "expected_roi" not in df.columns:
        if "model_score" not in df.columns:
            raise KeyError("Se requieren `ev_per_share` + `expected_roi` o `model_score` para generar señales.")
        buy_threshold = 0.60 if buy_threshold is None else buy_threshold
        strong_buy_threshold = 0.75 if strong_buy_threshold is None else strong_buy_threshold
        quality_mask = (
            (df["price_yes"] >= min_price_yes)
            & (df["price_yes"] <= max_price_yes)
            & (df.get("days_to_end", 0) <= max_days_to_end)
            & (df.get("liquidity", 0) >= min_liquidity)
            & (df.get("volume_24h", 0) >= min_volume_24h)
            & (df.get("spread", 0) <= max_spread)
        )
        df["expected_roi_capped"] = np.nan
        df.loc[quality_mask & (df["model_score"] >= buy_threshold), "signal"] = "BUY"
        df.loc[quality_mask & (df["model_score"] >= strong_buy_threshold), "signal"] = "STRONG BUY"
        df["signal_strength"] = np.clip(df["model_score"] / max(strong_buy_threshold, 1e-6), 0.0, 2.0)
        return df

    df["expected_roi_capped"] = np.clip(df["expected_roi"], -expected_roi_cap, expected_roi_cap)

    quality_mask = (
        (df["price_yes"] >= min_price_yes)
        & (df["price_yes"] <= max_price_yes)
        & (df["days_to_end"] <= max_days_to_end)
        & (df.get("liquidity", 0) >= min_liquidity)
        & (df.get("volume_24h", 0) >= min_volume_24h)
        & (df.get("spread", 0) <= max_spread)
    )

    buy_mask = (
        quality_mask
        & (df["ev_per_share"] >= buy_ev_threshold)
        & (df["expected_roi_capped"] >= buy_roi_threshold)
    )
    strong_buy_mask = (
        quality_mask
        & (df["ev_per_share"] >= strong_buy_ev_threshold)
        & (df["expected_roi_capped"] >= strong_buy_roi_threshold)
    )

    df.loc[buy_mask, "signal"] = "BUY"
    df.loc[strong_buy_mask, "signal"] = "STRONG BUY"
    df["signal_strength"] = np.clip(df["ev_per_share"] / max(strong_buy_ev_threshold, 1e-6), 0.0, 2.0)
    return df
