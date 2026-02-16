"""Limpieza y preprocesamiento de datos crudos de Polymarket."""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


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


def compute_time_features(
    df: pd.DataFrame,
    reference_time: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Calcula features temporales relativas a un punto de referencia.

    Args:
        df: DataFrame con columnas de fecha ya parseadas.
        reference_time: Timestamp de referencia. Si es None, usa el momento
            actual. Para entrenamiento, debe ser un timestamp anterior a la
            resolución del mercado para evitar data leakage.
    """
    df = df.copy()
    if reference_time is None:
        reference_time = pd.Timestamp.now(tz="UTC")

    if "endDate" in df.columns:
        df["days_to_resolution"] = (df["endDate"] - reference_time).dt.total_seconds() / 86400
        df["days_to_resolution"] = df["days_to_resolution"].clip(lower=0)

    if "createdAt" in df.columns:
        df["market_age_days"] = (reference_time - df["createdAt"]).dt.total_seconds() / 86400
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


def get_snapshot_price(
    market: dict,
    price_histories: dict | None = None,
    snapshot_offset_days: int = 7,
) -> float | None:
    """
    Obtiene el precio "Yes" en un snapshot temporal anterior a la resolución.

    Esto evita data leakage: en lugar de usar el precio final (que ya refleja
    la resolución), tomamos un precio de N días antes del cierre.

    Args:
        market: Diccionario del mercado.
        price_histories: Dict {market_id: [{"t": timestamp, "p": price}, ...]}.
        snapshot_offset_days: Cuántos días antes del endDate tomar el snapshot.

    Returns:
        Precio snapshot, o None si no se puede calcular.
    """
    market_id = market.get("id", "")

    # Si hay historial de precios, usarlo para obtener un snapshot real
    if price_histories and market_id in price_histories:
        history = price_histories[market_id]
        if isinstance(history, list) and len(history) > 0:
            end_date_str = market.get("endDate", "")
            if end_date_str:
                try:
                    end_date = pd.to_datetime(end_date_str, utc=True)
                    snapshot_time = end_date - pd.Timedelta(days=snapshot_offset_days)

                    # Buscar el precio más cercano al snapshot_time
                    best_price = None
                    best_delta = float("inf")
                    for point in history:
                        t = point.get("t", point.get("timestamp", 0))
                        p = point.get("p", point.get("price", 0))
                        try:
                            if isinstance(t, (int, float)):
                                point_time = pd.to_datetime(t, unit="s", utc=True)
                            else:
                                point_time = pd.to_datetime(t, utc=True)
                            # Solo considerar puntos anteriores al snapshot
                            delta = abs((point_time - snapshot_time).total_seconds())
                            if point_time <= snapshot_time and delta < best_delta:
                                best_delta = delta
                                best_price = float(p)
                        except (ValueError, TypeError):
                            continue

                    if best_price is not None and 0 < best_price < 1:
                        return best_price
                except (ValueError, TypeError):
                    pass

    # Fallback: usar outcomePrices pero solo si el mercado no está resuelto,
    # o si no hay historial disponible. Marcar como potencialmente contaminado.
    outcome_prices = market.get("outcomePrices", [])
    if isinstance(outcome_prices, str):
        try:
            outcome_prices = json.loads(outcome_prices)
        except (json.JSONDecodeError, TypeError):
            return None

    if outcome_prices:
        price = float(outcome_prices[0])
        # Para mercados resueltos, el precio final está en ~1.0 o ~0.0,
        # lo que NO es un buen snapshot. Solo usar si parece razonable.
        if 0.05 < price < 0.95:
            return price
        # Precio extremo en mercado resuelto = dato contaminado, descartar
        if market.get("resolved") or market.get("closed"):
            logger.warning(
                "Market %s: precio final extremo (%.2f), descartando. "
                "Se necesita price_history para un snapshot válido.",
                market_id, price,
            )
            return None
        return price

    return None


def compute_label(
    market: dict,
    snapshot_price_yes: float,
    profit_threshold: float = 0.05,
) -> int:
    """
    Calcula la label para un mercado resuelto.

    Usa un profit_threshold en lugar del umbral fijo de 0.95 para que
    el criterio sea relativo al precio de snapshot, no absoluto.

    Args:
        market: Diccionario del mercado resuelto.
        snapshot_price_yes: Precio "Yes" en el momento del snapshot.
        profit_threshold: Retorno mínimo para considerar "rentable".

    Returns:
        1 si comprar "Yes" hubiera dado un retorno >= profit_threshold
        0 si no fue rentable
        -1 si resolución ambigua (descartar)
    """
    resolution = str(market.get("resolution", "")).strip().lower()
    if resolution == "yes":
        # Retorno: (1.0 - precio_compra) / precio_compra
        if snapshot_price_yes > 0:
            expected_return = (1.0 - snapshot_price_yes) / snapshot_price_yes
            return 1 if expected_return >= profit_threshold else 0
        return 0
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


def preprocess_markets(
    filepath: str,
    reference_time: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Pipeline completo de preprocesamiento.

    Args:
        filepath: Ruta al JSON de mercados.
        reference_time: Timestamp de referencia para features temporales.
            None = momento actual (adecuado para scoring en vivo).
    """
    df = load_raw_markets(filepath)
    df = parse_json_fields(df)
    df = extract_prices(df)
    df = parse_dates(df)
    df = compute_time_features(df, reference_time=reference_time)
    df = clean_numeric_columns(df)
    df = compute_derived_features(df)
    return df


def preprocess_market_dict(
    market: dict,
    reference_time: pd.Timestamp | None = None,
) -> dict:
    """
    Preprocesa un mercado individual (dict) añadiendo features temporales
    y derivadas. Útil cuando se trabaja con listas de dicts en lugar de
    DataFrames.

    Args:
        market: Diccionario del mercado (ya con JSON fields parseados).
        reference_time: Timestamp de referencia.

    Returns:
        Dict con campos adicionales: days_to_resolution, market_age_days,
        spread, volume_liquidity_ratio, price_yes, price_no.
    """
    if reference_time is None:
        reference_time = pd.Timestamp.now(tz="UTC")

    m = market.copy()

    # Parse JSON fields si necesario
    for field in ["outcomes", "outcomePrices", "clobTokenIds"]:
        val = m.get(field)
        if isinstance(val, str):
            try:
                m[field] = json.loads(val)
            except (json.JSONDecodeError, TypeError):
                m[field] = []

    # Precios
    outcome_prices = m.get("outcomePrices", [])
    if outcome_prices:
        m["price_yes"] = float(outcome_prices[0])
        m["price_no"] = float(outcome_prices[1]) if len(outcome_prices) > 1 else 1 - m["price_yes"]

    # Numéricas
    for col in ["volume", "volume24hr", "liquidity", "bestBid", "bestAsk", "spread"]:
        try:
            m[col] = float(m[col]) if m.get(col) is not None else 0.0
        except (ValueError, TypeError):
            m[col] = 0.0

    # Temporales
    for col in ["endDate", "createdAt"]:
        val = m.get(col)
        if val and not isinstance(val, pd.Timestamp):
            try:
                m[col] = pd.to_datetime(val, utc=True)
            except (ValueError, TypeError):
                m[col] = None

    end_date = m.get("endDate")
    if end_date and isinstance(end_date, pd.Timestamp):
        m["days_to_resolution"] = max(0.0, (end_date - reference_time).total_seconds() / 86400)
    else:
        m["days_to_resolution"] = 30.0

    created = m.get("createdAt")
    if created and isinstance(created, pd.Timestamp):
        m["market_age_days"] = max(0.0, (reference_time - created).total_seconds() / 86400)
    else:
        m["market_age_days"] = 0.0

    # Derivadas
    if m["spread"] == 0 and m["bestAsk"] > 0:
        m["spread"] = m["bestAsk"] - m["bestBid"]
    m["volume_liquidity_ratio"] = (
        m["volume24hr"] / m["liquidity"] if m["liquidity"] > 0 else 0.0
    )

    return m
