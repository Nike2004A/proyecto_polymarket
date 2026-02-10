"""Limpieza y preprocesamiento de datos crudos de Polymarket."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def load_raw_markets(filepath: str) -> pd.DataFrame:
    """Carga mercados crudos desde JSON y devuelve un DataFrame."""
    path = Path(filepath)
    with open(path, "r", encoding="utf-8") as f:
        markets = json.load(f)
    return pd.DataFrame(markets)


def parse_json_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Parsea campos que vienen como JSON strings."""
    df = df.copy()
    for col in ["outcomes", "outcomePrices", "clobTokenIds"]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: json.loads(x) if isinstance(x, str) else (x if isinstance(x, list) else [])
            )
    return df


def extract_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Extrae price_yes y price_no de outcomePrices."""
    df = df.copy()
    df["price_yes"] = df["outcomePrices"].apply(
        lambda x: float(x[0]) if isinstance(x, list) and len(x) > 0 else np.nan
    )
    df["price_no"] = df["outcomePrices"].apply(
        lambda x: float(x[1]) if isinstance(x, list) and len(x) > 1 else np.nan
    )
    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte columnas de fecha a datetime."""
    df = df.copy()
    for col in ["startDate", "endDate", "createdAt"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    return df


def compute_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula features temporales."""
    df = df.copy()
    now = pd.Timestamp.now(tz="UTC")

    if "endDate" in df.columns:
        df["days_to_resolution"] = (df["endDate"] - now).dt.total_seconds() / 86400
        df["days_to_resolution"] = df["days_to_resolution"].clip(lower=0)

    if "createdAt" in df.columns:
        df["market_age_days"] = (now - df["createdAt"]).dt.total_seconds() / 86400
        df["market_age_days"] = df["market_age_days"].clip(lower=0)

    return df


def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia y convierte columnas numéricas."""
    df = df.copy()
    numeric_cols = [
        "volume", "volume24hr", "liquidity",
        "bestBid", "bestAsk", "spread",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula features derivadas."""
    df = df.copy()

    # Spread (si no existe, calcularlo de bestAsk - bestBid)
    if "spread" not in df.columns or df["spread"].isna().all():
        df["spread"] = df.get("bestAsk", 0) - df.get("bestBid", 0)

    # Volume/Liquidity ratio
    df["volume_liquidity_ratio"] = np.where(
        df["liquidity"] > 0,
        df["volume24hr"] / df["liquidity"],
        0,
    )

    # Neg risk como feature binaria
    if "negRisk" in df.columns:
        df["neg_risk"] = df["negRisk"].astype(int)
    else:
        df["neg_risk"] = 0

    return df


def compute_label(market: dict, snapshot_price_yes: float) -> int:
    """
    Calcula la label para un mercado resuelto.

    Returns:
        1 si comprar "Yes" hubiera sido rentable
        0 si no
        -1 si resolución ambigua (descartar)
    """
    resolution = str(market.get("resolution", "")).strip().lower()
    if resolution == "yes":
        return 1 if snapshot_price_yes < 0.95 else 0
    elif resolution == "no":
        return 0
    else:
        return -1


def compute_continuous_label(market: dict, snapshot_price_yes: float) -> float | None:
    """
    Label continua: retorno esperado.
    payout - precio_de_compra
    """
    resolution = str(market.get("resolution", "")).strip().lower()
    if resolution == "yes":
        return 1.0 - snapshot_price_yes
    elif resolution == "no":
        return -snapshot_price_yes
    return None


def preprocess_markets(filepath: str) -> pd.DataFrame:
    """Pipeline completo de preprocesamiento."""
    df = load_raw_markets(filepath)
    df = parse_json_fields(df)
    df = extract_prices(df)
    df = parse_dates(df)
    df = compute_time_features(df)
    df = clean_numeric_columns(df)
    df = compute_derived_features(df)
    return df
