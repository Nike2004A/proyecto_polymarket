# Polymarket ML Trading Signal Analyzer

Sistema end-to-end en Python que consume la API de Polymarket, extrae features de mercados de predicción, entrena un modelo con PyTorch para identificar oportunidades de compra (mercados infravalorados) y presenta los resultados en Jupyter Notebooks interactivos con visualizaciones.

## Arquitectura del Pipeline

```
┌──────────────────────────────────────────────────────────────────────┐
│                        PIPELINE COMPLETO                             │
│                                                                      │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────────────────┐  │
│  │  DATA LAYER │───>│ FEATURE ENG. │───>│   MODEL (PyTorch)      │  │
│  │             │    │              │    │                        │  │
│  │ - Gamma API │    │ - Precio     │    │ - MarketValueNet       │  │
│  │ - CLOB API  │    │ - Volumen    │    │   (Wide & Deep)        │  │
│  │ - Data API  │    │ - Liquidez   │    │                        │  │
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

```
polymarket-ml-analyzer/
├── config/
│   └── config.yaml                # Hiperparámetros y configuración general
├── src/
│   ├── data/
│   │   ├── client.py              # Cliente unificado (Gamma + CLOB + Data API)
│   │   ├── fetcher.py             # Descarga masiva con cache local
│   │   └── preprocessing.py       # Limpieza, parseo de JSON, cálculo de labels
│   ├── features/
│   │   ├── numerical.py           # 14 features numéricas
│   │   ├── categorical.py         # CategoryEncoder (nn.Embedding)
│   │   ├── text.py                # Sentence Transformers embeddings
│   │   └── pipeline.py            # Pipeline completo: raw market -> tensors
│   ├── model/
│   │   ├── architecture.py        # MarketValueNet (Wide & Deep)
│   │   ├── dataset.py             # PolymarketDataset + DataLoaders
│   │   ├── train.py               # Training loop (AdamW + CosineAnnealing)
│   │   └── evaluate.py            # Métricas, confusion matrix, backtesting
│   └── scoring/
│       ├── scorer.py              # Scoring de mercados activos
│       └── signals.py             # Generación de señales de compra
├── notebooks/
│   ├── 01_data_exploration.ipynb  # EDA de mercados de Polymarket
│   ├── 02_feature_engineering.ipynb # Construcción y análisis de features
│   ├── 03_model_training.ipynb    # Entrenamiento y evaluación del modelo
│   └── 04_live_scoring.ipynb      # Scoring en vivo + dashboard + backtesting
├── data/
│   ├── raw/                       # Datos crudos de la API (JSON)
│   ├── processed/                 # Features procesadas (npy)
│   └── models/                    # Checkpoints del modelo (.pt)
├── figures/                       # Gráficas exportadas
├── requirements.txt
└── setup.py
```

## Instalacion

```bash
# Clonar el repositorio
git clone <repo_url>
cd polymarket-ml-analyzer

# Crear entorno virtual (recomendado)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Instalar dependencias
pip install -r requirements.txt
```

### Dependencias principales

| Paquete | Uso |
|---|---|
| `torch` | Modelo MarketValueNet (Wide & Deep) |
| `pandas`, `numpy` | Manipulación de datos |
| `scikit-learn` | Métricas, scaler, train/val split |
| `sentence-transformers` | Embeddings semánticos de preguntas |
| `requests` | Consumo de APIs de Polymarket |
| `matplotlib`, `seaborn`, `plotly` | Visualizaciones |
| `jupyter` | Notebooks interactivos |

## Uso

### Ejecucion completa via CLI

```bash
# 1. Descargar datos de Polymarket (mercados activos + resueltos + order books)
python -m src.data.fetcher --mode full --output data/raw/

# 2. Generar features (numéricas + categorías + text embeddings)
python -m src.features.pipeline --input data/raw/ --output data/processed/

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

1. **01_data_exploration** - Descarga datos, EDA, distribuciones de precios, volumen y liquidez
2. **02_feature_engineering** - Extracción de features, correlaciones, distribución por clase
3. **03_model_training** - Entrenamiento del modelo, curvas de loss/accuracy, confusion matrix, ROC
4. **04_live_scoring** - Scoring en vivo, señales de compra, dashboard visual, backtesting

## Modelo: MarketValueNet

Arquitectura **Wide & Deep** para datos tabulares mixtos:

```
                    ┌──────────────────────┐
                    │   numerical (14)     │
                    │   + category emb (8) │──> Deep: [256, 128, 64] ──┐
                    │   + text emb (384)   │    (BN + ReLU + Dropout)  │
                    └──────────────────────┘                           │
                                                                       ├──> Head ──> Score
                    ┌──────────────────────┐                           │
                    │   numerical (14)     │──> Wide: Linear(14, 32) ──┘
                    └──────────────────────┘
```

### Features de entrada (14 numericas)

| Feature | Descripcion |
|---|---|
| `price_yes` / `price_no` | Probabilidad implicita del mercado |
| `spread` | Spread bid-ask (ineficiencia del mercado) |
| `volume_24h` / `volume_total` | Actividad de trading |
| `liquidity` | Profundidad del mercado |
| `volume_liquidity_ratio` | Velocidad de rotacion |
| `days_to_resolution` | Tiempo restante hasta resolucion |
| `market_age_days` | Madurez del mercado |
| `price_momentum_7d` | Tendencia reciente del precio |
| `price_volatility_7d` | Incertidumbre |
| `bid_depth` / `ask_depth` | Soporte de compra / presion de venta |
| `book_imbalance` | Señal de direccion del order book |

### Label de entrenamiento

Para mercados resueltos:
- **Label = 1 (Buy)**: El mercado resolvio "Yes" y el precio de compra era < 0.95
- **Label = 0 (No Buy)**: El mercado resolvio "No"

### Señales de salida

| Señal | Condicion |
|---|---|
| **STRONG BUY** | Score >= 0.75, alpha >= 0.05, buena liquidez |
| **BUY** | Score >= 0.60, alpha >= 0.05 |
| **HOLD** | No cumple filtros de calidad o score bajo |

## APIs de Polymarket

El proyecto consume tres APIs publicas:

| API | Base URL | Uso |
|---|---|---|
| **Gamma** | `https://gamma-api.polymarket.com` | Metadata de mercados, eventos, tags |
| **CLOB** | `https://clob.polymarket.com` | Order book, precios en tiempo real |
| **Data** | `https://data-api.polymarket.com` | Series temporales historicas |

No se requiere autenticacion para lectura de datos.

## Disclaimer

Este proyecto es con fines educativos y de investigacion. Trading en mercados de prediccion conlleva riesgo financiero. El modelo no garantiza ganancias. Siempre haz tu propia investigacion antes de tomar decisiones financieras.
