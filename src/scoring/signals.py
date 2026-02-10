"""Generación de señales de compra/venta basadas en el scoring del modelo."""

import pandas as pd
import numpy as np


def generate_signals(
    scored_df: pd.DataFrame,
    buy_threshold: float = 0.6,
    strong_buy_threshold: float = 0.75,
    min_liquidity: float = 1000.0,
    min_volume_24h: float = 100.0,
    max_spread: float = 0.10,
    min_alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Genera señales de compra a partir del DataFrame de mercados puntuados.

    Args:
        scored_df: DataFrame con columnas model_score, price_yes, liquidity, etc.
        buy_threshold: Umbral mínimo para señal de compra.
        strong_buy_threshold: Umbral para señal de compra fuerte.
        min_liquidity: Liquidez mínima requerida.
        min_volume_24h: Volumen 24h mínimo.
        max_spread: Spread máximo aceptable.
        min_alpha: Alpha mínimo esperado.

    Returns:
        DataFrame con señales añadidas.
    """
    df = scored_df.copy()

    # Señal base
    df["signal"] = "HOLD"

    # Filtros de calidad
    quality_mask = (
        (df["liquidity"] >= min_liquidity)
        & (df["volume_24h"] >= min_volume_24h)
        & (df["spread"] <= max_spread)
    )

    # Señales de compra
    buy_mask = (
        quality_mask
        & (df["model_score"] >= buy_threshold)
        & (df["expected_alpha"] >= min_alpha)
    )
    strong_buy_mask = buy_mask & (df["model_score"] >= strong_buy_threshold)

    df.loc[buy_mask, "signal"] = "BUY"
    df.loc[strong_buy_mask, "signal"] = "STRONG BUY"

    # Confianza (normalizada entre 0-1)
    df["confidence"] = np.clip(
        (df["model_score"] - buy_threshold) / (1 - buy_threshold),
        0, 1,
    )

    # Razón de la señal
    df["signal_reason"] = ""
    df.loc[strong_buy_mask, "signal_reason"] = "Alto score + alpha positivo + buena liquidez"
    df.loc[buy_mask & ~strong_buy_mask, "signal_reason"] = "Score aceptable + alpha positivo"
    df.loc[~quality_mask, "signal_reason"] = "Filtrado por calidad (liquidez/volumen/spread)"

    # Tamaño de posición sugerido (Kelly simplificado)
    df["suggested_position_pct"] = 0.0
    df.loc[buy_mask, "suggested_position_pct"] = np.clip(
        df.loc[buy_mask, "expected_alpha"] * df.loc[buy_mask, "confidence"] * 100,
        1.0,
        10.0,
    )

    # URL de Polymarket
    df["polymarket_url"] = df["slug"].apply(
        lambda s: f"https://polymarket.com/event/{s}" if s else ""
    )

    return df


def format_signals_report(signals_df: pd.DataFrame) -> str:
    """Genera un reporte formateado de señales."""
    lines = []
    lines.append("=" * 80)
    lines.append("POLYMARKET TRADING SIGNALS REPORT")
    lines.append("=" * 80)

    # Resumen
    signal_counts = signals_df["signal"].value_counts()
    lines.append(f"\nResumen de señales:")
    for signal, count in signal_counts.items():
        lines.append(f"  {signal}: {count}")

    # Strong buys
    strong_buys = signals_df[signals_df["signal"] == "STRONG BUY"]
    if not strong_buys.empty:
        lines.append(f"\n{'─'*40}")
        lines.append("STRONG BUY Signals:")
        lines.append(f"{'─'*40}")
        for _, row in strong_buys.iterrows():
            lines.append(f"\n  {row['question'][:70]}")
            lines.append(
                f"    Score: {row['model_score']:.3f} | "
                f"Precio: ${row['price_yes']:.2f} | "
                f"Alpha: {row['expected_alpha']:.3f}"
            )
            lines.append(
                f"    Confianza: {row['confidence']:.1%} | "
                f"Posición sugerida: {row['suggested_position_pct']:.1f}%"
            )

    # Regular buys
    buys = signals_df[signals_df["signal"] == "BUY"]
    if not buys.empty:
        lines.append(f"\n{'─'*40}")
        lines.append("BUY Signals:")
        lines.append(f"{'─'*40}")
        for _, row in buys.iterrows():
            lines.append(f"\n  {row['question'][:70]}")
            lines.append(
                f"    Score: {row['model_score']:.3f} | "
                f"Precio: ${row['price_yes']:.2f} | "
                f"Alpha: {row['expected_alpha']:.3f}"
            )

    lines.append(f"\n{'='*80}")
    return "\n".join(lines)
