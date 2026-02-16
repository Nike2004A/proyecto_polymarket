"""Generación de señales de compra/venta basadas en el scoring del modelo."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_signals(
    scored_df: pd.DataFrame,
    buy_threshold: float = 0.6,
    strong_buy_threshold: float = 0.75,
    min_liquidity: float = 1000.0,
    min_volume_24h: float = 100.0,
    max_spread: float = 0.10,
) -> pd.DataFrame:
    """
    Genera señales de compra a partir del DataFrame de mercados puntuados.

    NOTA IMPORTANTE: Las señales se basan en el model_score, que es un score
    de clasificación NO calibrado. No debe interpretarse como probabilidad
    real de profit. Las señales son indicativas y requieren análisis adicional.

    Args:
        scored_df: DataFrame con columnas model_score, price_yes, liquidity, etc.
        buy_threshold: Umbral mínimo de score para señal de compra.
        strong_buy_threshold: Umbral de score para señal fuerte.
        min_liquidity: Liquidez mínima requerida.
        min_volume_24h: Volumen 24h mínimo.
        max_spread: Spread máximo aceptable.

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

    filtered_out = (~quality_mask).sum()
    if filtered_out > 0:
        logger.info(
            "%d/%d mercados filtrados por calidad (liquidez/volumen/spread).",
            filtered_out, len(df),
        )

    # Señales de compra (basadas solo en score, sin "alpha" no calibrado)
    buy_mask = quality_mask & (df["model_score"] >= buy_threshold)
    strong_buy_mask = buy_mask & (df["model_score"] >= strong_buy_threshold)

    df.loc[buy_mask, "signal"] = "BUY"
    df.loc[strong_buy_mask, "signal"] = "STRONG BUY"

    # Confianza relativa (normalizada entre 0-1, útil solo para ranking)
    df["confidence"] = np.clip(
        (df["model_score"] - buy_threshold) / (1 - buy_threshold),
        0, 1,
    )

    # Razón de la señal
    df["signal_reason"] = ""
    df.loc[strong_buy_mask, "signal_reason"] = "Score alto + buena liquidez"
    df.loc[buy_mask & ~strong_buy_mask, "signal_reason"] = "Score aceptable"
    df.loc[~quality_mask, "signal_reason"] = "Filtrado por calidad (liquidez/volumen/spread)"

    # Tamaño de posición sugerido (conservador, basado en confianza relativa)
    df["suggested_position_pct"] = 0.0
    df.loc[buy_mask, "suggested_position_pct"] = np.clip(
        df.loc[buy_mask, "confidence"] * 5.0,  # max 5% del capital
        1.0,
        5.0,
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
    lines.append(
        "\nDISCLAIMER: Estas señales se basan en un modelo de clasificación"
    )
    lines.append(
        "no calibrado. NO representan probabilidades reales ni retornos"
    )
    lines.append(
        "esperados. Usar solo como punto de partida para análisis propio."
    )

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
                f"Precio: ${row['price_yes']:.2f}"
            )
            lines.append(
                f"    Confianza (relativa): {row['confidence']:.1%} | "
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
                f"Precio: ${row['price_yes']:.2f}"
            )

    lines.append(f"\n{'='*80}")
    return "\n".join(lines)
