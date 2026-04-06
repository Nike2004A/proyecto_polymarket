# Polymarket ML Trading Signal Analyzer

Sistema end-to-end en Python que consume la API de Polymarket, extrae features de mercados de predicción, entrena un modelo con PyTorch para identificar oportunidades de compra (mercados infravalorados) y presenta los resultados en Jupyter Notebooks interactivos con visualizaciones.

## Estado del pipeline

```
[✓] Ingesta de datos      — src/data/ + notebooks/00, 01
[✓] Feature engineering   — src/features/ + data/processed/
[ ] Entrenamiento         — src/model/ + notebook 04
[ ] Scoring en vivo       — src/scoring/ + notebook 05
```

## Arquitectura del Pipeline

```text
┌──────────────────────────────────────────────────────────────────────┐
│                        PIPELINE COMPLETO                             │
│                                                                      │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────────────────┐  │
│  │  DATA LAYER │───>│ FEATURE ENG. │───>│   MODEL (PyTorch)      │  │
│  │             │    │              │    │                        │  │
│  │ - Gamma API │    │ - Precio     │    │ - MarketValueNet       │  │
│  │ - CLOB API  │    │ - Volumen    │    │   (Wide & Deep)        │  │
│  │             │    │ - Liquidez   │    │                        │  │
│  └─────────────┘    │ - Momentum   │    │ - Entrenamiento con    │  │
│                     │ - Categoría  │    │   mercados resueltos   │  │
│                     │ - Spread     │    │   (labels reales)      │  │
│                     │ - Text Emb.  │    └───────────┬────────────┘  │
│                     └──────────────┘                │               │
│                     ┌───────────────────────────────▼────────────┐  │
│                     │         SCORING & OUTPUT                    │  │
│                     │                                            │  │
│                     │ - Ranking de mercados por alpha esperado   │  │
│                     │ - Señales: STRONG BUY / BUY / HOLD        │  │
│                     │ - Jupyter Notebooks con visualizaciones    │  │
│                     └────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

## Estructura del Proyecto

```text
polymarket-ml-analyzer/
├── config/
│   └── config.yaml                # Hiperparámetros y configuración general
├── src/
│   ├── data/
│   │   ├── client.py              # Cliente unificado (Gamma + CLOB API)
│   │   ├── fetcher.py             # Descarga masiva incremental con cache local
│   │   └── preprocessing.py       # Limpieza, parseo de JSON, cálculo de labels
│   ├── features/
│   │   ├── numerical.py           # 23 features numéricas
│   │   ├── categorical.py         # CategoryEncoder — 10 categorías via slug/keyword
│   │   ├── text.py                # Sentence Transformers embeddings (MiniLM-L6)
│   │   └── pipeline.py            # FeaturePipeline: raw markets -> .npy tensors
│   ├── model/
│   │   ├── architecture.py        # MarketValueNet (Wide & Deep)
│   │   ├── dataset.py             # PolymarketDataset + temporal DataLoaders
│   │   ├── train.py               # Training loop (AdamW + CosineAnnealing)
│   │   └── evaluate.py            # Métricas, confusion matrix, backtesting
│   └── scoring/
│       ├── scorer.py              # Scoring de mercados activos
│       └── signals.py             # Generación de señales de compra
├── notebooks/
│   ├── 00_playground.ipynb             # Sandbox / exploración libre
│   ├── 01_data_exploration.ipynb       # EDA completo — 18 secciones
│   ├── 02_feature_engineering.ipynb    # Construcción y análisis de features
│   ├── 03_processed_dataset_eda.ipynb  # EDA sobre el dataset procesado (.npy)
│   ├── 04_model_training.ipynb         # Entrenamiento, curvas, confusion matrix, ROC
│   └── 05_live_scoring.ipynb           # Scoring en vivo, señales, dashboard, backtesting
├── data/
│   ├── raw/                       # Datos crudos de la API (JSON)
│   │   ├── active_markets.json    #   900 activos (todos con historial + order book)
│   │   ├── resolved_markets.json  #   24,000 resueltos (todos con historial)
│   │   ├── price_histories.json   #   24,900 historiales (union exacta, sin solapamiento)
│   │   ├── order_books.json       #   900 order books (uno por activo)
│   │   └── fetch_metadata.json    #   Stats del último fetch
│   ├── processed/                 # Features procesadas (.npy) — generadas por el pipeline
│   └── models/                    # Checkpoints del modelo (.pt)
├── figures/                       # Gráficas exportadas desde notebooks
├── requirements.txt
└── setup.py
```

## Instalacion

```bash
git clone <repo_url>
cd polymarket-ml-analyzer

python -m venv .venv
source .venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

### Dependencias principales

| Paquete | Uso |
|---|---|
| `torch` | Modelo MarketValueNet (Wide & Deep) |
| `pandas`, `numpy` | Manipulación de datos |
| `scikit-learn` | Métricas, scaler |
| `sentence-transformers` | Embeddings semánticos de preguntas (MiniLM-L6-v2) |
| `requests` | Consumo de APIs de Polymarket |
| `matplotlib`, `seaborn`, `plotly` | Visualizaciones |
| `jupyter` | Notebooks interactivos |

## Uso

### Ejecucion completa via CLI

```bash
# 1. Descargar datos (incremental, retomable si se interrumpe)
caffeinate -i python -m src.data.fetcher --mode full

# Reanudar solo historiales si se interrumpió:
python -m src.data.fetcher --mode histories

# 2. Generar features (produce data/processed/*.npy)
python -m src.features.pipeline

# 3. Entrenar modelo
python -m src.model.train --data-dir data/processed --epochs 50 --batch-size 64

# 4. Scoring de mercados activos (top 20 oportunidades)
python -m src.scoring.scorer --top 20
```

### Ejecucion via Notebooks

```bash
jupyter notebook notebooks/
```

Los notebooks están diseñados para ejecutarse en orden:

1. **00_playground** — Sandbox para exploración libre
2. **01_data_exploration** — EDA completo (18 secciones): calidad temporal, calibración, favorite-longshot bias, trayectorias, análisis temporal, separabilidad de features
3. **02_feature_engineering** — Construcción de features, correlaciones, distribución por clase
4. **03_processed_dataset_eda** — EDA sobre el dataset procesado (.npy)
5. **04_model_training** — Entrenamiento, curvas de loss/AUC, confusion matrix, ROC
6. **05_live_scoring** — Scoring en vivo, señales de compra, dashboard, backtesting

## Dataset procesado

El pipeline de features produce los siguientes archivos en `data/processed/`:

| Archivo | Shape | Descripción |
|---|---|---|
| `numerical_features.npy` | (22478, 23) | Features numéricas normalizadas (StandardScaler) |
| `text_embeddings.npy` | (22478, 384) | Embeddings MiniLM-L6-v2 de la pregunta |
| `category_ids.npy` | (22478,) | ID de categoría (0–9) |
| `labels.npy` | (22478,) | 1 = buy, 0 = no buy |
| `end_dates.npy` | (22478,) | Timestamps para temporal split |
| `pipeline/` | — | Scaler + encoders serializados |

De los 24,000 mercados resueltos descargados, 22,478 producen features válidas (93.7%). Los 1,522 descartados son: 834 sin snapshot anti-leakage válido + 688 con resolución ambigua.

## Modelo: MarketValueNet

Arquitectura **Wide & Deep** para datos tabulares mixtos:

```text
                    ┌──────────────────────┐
                    │   numerical (23)     │
                    │   + category emb (8) │──> Deep: [256, 128, 64] ──┐
                    │   + text emb (384)   │    (BN + ReLU + Dropout)  │
                    └──────────────────────┘                           │
                                                                       ├──> Head ──> Score
                    ┌──────────────────────┐                           │
                    │   numerical (23)     │──> Wide: Linear(23, 32) ──┘
                    └──────────────────────┘
```

### Features de entrada (23 numericas)

| Grupo | Feature | Descripcion |
|---|---|---|
| Precio snapshot | `price_yes`, `price_no`, `spread` | Probabilidad implícita y spread bid-ask |
| Volumen/liquidez | `volume_24h`, `volume_total`, `liquidity`, `volume_liquidity_ratio` | Actividad y profundidad del mercado |
| Temporales | `days_to_resolution`, `market_age_days` | Tiempo restante y madurez |
| TS momentum | `price_momentum_7d/14d/30d`, `price_volatility_7d/30d` | Tendencia y volatilidad histórica |
| TS tendencia | `price_trend_slope`, `ewm_momentum`, `ts_coverage`, `ts_days_span` | Dirección de largo plazo |
| Order book | `bid_depth`, `ask_depth`, `book_imbalance` | Soporte y presión del libro (solo activos en scoring) |
| Trayectoria | `price_at_halflife` | Precio a mitad de vida del mercado |
| Estructural | `neg_risk` | negRisk=True → resuelve Yes 14% vs 43% en mercados normales |

> **Nota**: Las 5 features de order book (`bid_depth`, `ask_depth`, `book_imbalance`, `liquidity`, `volume_liquidity_ratio`) son cero en todos los mercados resueltos (CLOB cerrado al resolver). Son útiles exclusivamente en scoring sobre activos.

### Label de entrenamiento

Para cada mercado resuelto se extrae el precio en un **snapshot anti-leakage**:

- **Snapshot primario**: precio 7 días antes del `endDate` (desde el historial de precios).
- **Snapshot adaptivo**: si el mercado vivió < 7 días, se usa el último precio disponible antes del `endDate` con `0 < precio < 1`. Esto recupera ~13,500 mercados de vida corta sin introducir leakage.

**Regla de label:**
- **Label = 1 (Buy)**: mercado resolvió "Yes" y el retorno esperado `(1 − precio_snapshot) / precio_snapshot ≥ 5%`
- **Label = 0 (No Buy)**: resolvió "No", o el retorno esperado fue insuficiente

**Distribución resultante**: 33.1% positivos (buy=1), 66.9% negativos.

### Señales de salida

| Señal | Condicion |
|---|---|
| **STRONG BUY** | Score >= 0.75, alpha >= 0.05, buena liquidez |
| **BUY** | Score >= 0.60, alpha >= 0.05 |
| **HOLD** | No cumple filtros de calidad o score bajo |

## Hallazgos del EDA

Del análisis en `notebooks/01_data_exploration.ipynb`:

- **Calibración**: Los precios de Polymarket están bien calibrados — la curva de calibración sigue de cerca la diagonal (MAE ≈ 0.03).
- **Favorite-Longshot Bias**: Los longshots (precio < 0.15) están ligeramente subvalorados en promedio; los favoritos (precio > 0.85) están ligeramente sobrevalorados.
- **negRisk**: Feature crítica. Mercados con `negRisk=True` resuelven Yes el 14.1% vs 43.1% en mercados normales — diferencia de 3×.
- **Calidad temporal**: Cero mercados con gaps > 48h en el historial de precios. Span mediano de 11.7 días.
- **Features TS**: `ewm_momentum` y `price_trend_slope` son las features con mayor separabilidad estadística (Mann-Whitney p < 0.001). `price_at_halflife` tiene correlación -0.82 con `price_yes` final, indicando convergencia de precios.

## APIs de Polymarket

El proyecto consume dos APIs publicas:

| API | Base URL | Uso |
|---|---|---|
| **Gamma** | `https://gamma-api.polymarket.com` | Metadata de mercados, resolución, tags |
| **CLOB** | `https://clob.polymarket.com` | Order book, historiales de precios |

No se requiere autenticacion. Rate limit: 0.2 s entre llamadas.

### Tamaños del dataset

| Conjunto | Mercados | Condición |
|---|---|---|
| Activos | 900 | con historial de precios + order book |
| Resueltos | 24,000 | con historial de precios completo |
| Historiales | 24,900 | unión exacta (sin solapamiento) |
| Order books | 900 | uno por mercado activo |
| **Training set** | **22,478** | resueltos con snapshot válido + label claro |

## Disclaimer

Este proyecto es con fines educativos y de investigacion. Trading en mercados de prediccion conlleva riesgo financiero. El modelo no garantiza ganancias. Siempre haz tu propia investigacion antes de tomar decisiones financieras.
