# Polymarket Snapshot Modeling

Pipeline reproducible para modelar mercados de Polymarket sin descargar datos nuevos. El sistema entrena tres modelos sobre snapshots históricos de mercados resueltos:

- un MLP tabular que predice el residual `label_yes - price_yes`
- un GRU secuencial sobre una rejilla temporal fija, también sobre residual
- un `HistGradientBoostingClassifier` tabular que predice `p_yes` directo

Los dos modelos residuales producen una estimación de `p_yes` derivada de:

- `p_yes_hat = clip(price_yes + residual_pred, 0, 1)`

El modelo de boosting predice `p_yes` directo y compite explícitamente contra el benchmark `price_yes`.

Importante: el modelo primario actual no es XGBoost. Es `HistGradientBoostingClassifier` de `scikit-learn`. Se eligió porque:

- entrena rápido con este dataset
- maneja bien no linealidades tabulares
- fue el único que terminó superando al benchmark de mercado en `Brier` y `log_loss` sobre el split actual
- deja un pipeline más simple y reproducible, sin depender de `xgboost` o `lightgbm`

La decisión operativa no sale del modelo directo, sino de una capa explícita de valor esperado:

- `ev_per_share = p_yes_calibrated - price_yes`
- `expected_roi = ev_per_share / price_yes`

El benchmark obligatorio es el propio mercado:

- `market baseline = p_hat = price_yes`

## Qué cambió

El sistema anterior mezclaba probabilidad del evento con edge económico y usaba features que no existían históricamente para train. Esta reconstrucción corrige eso:

- target nuevo de entrenamiento: `target_residual = label_yes - snapshot_price_yes`
- múltiples snapshots por mercado en horizontes fijos `1/3/7/14/30` días
- split temporal agrupado por `market_id`
- features tabulares observables en vivo
- ventanas por calendario real, no por número de ticks
- secuencias en grilla fija de `30 días x 12 horas = 60 pasos`
- filtros de precio antes de generar señales
- `expected_roi_capped` para evitar explosiones artificiales en mercados casi a cero
- métricas por bucket de horizonte `1-3d`, `4-14d`, `15d+`
- selección automática del modelo primario por utilidad out-of-sample

## Flujo

```text
data/raw/*.json
    -> src.features.pipeline
    -> data/processed + data/processed_ts
    -> src.model.train
    -> data/models/{market_value_baseline,price_sequence_gru,hist_gradient_boosting}
    -> src.scoring.scorer
    -> p_yes_hat + EV + signals
```

## Features permitidas

El set tabular final es snapshot-compatible:

- `snapshot_price_yes`
- `days_to_end`
- `market_age_days_at_snapshot`
- `days_since_last_trade`
- `history_points_1d/3d/7d/30d`
- `history_span_days`
- `return_1d/3d/7d/14d/30d`
- `realized_vol_1d/3d/7d/14d/30d`
- `trend_slope_7d/30d`
- `price_percentile_30d`
- `distance_to_30d_min`
- `distance_to_30d_max`
- `neg_risk`
- `category_id`
- `question_embedding`

## Features prohibidas

No entran al modelo porque no son reconstruibles con el mismo contrato histórico:

- `price_no`
- `volume_24h`
- `volume_total`
- `liquidity`
- `volume_liquidity_ratio`
- `bestBid`, `bestAsk`, `spread`
- `bid_depth`, `ask_depth`, `book_imbalance`

Esas variables pueden seguir existiendo como filtros downstream de ejecución, pero no como inputs del modelo.

## Modelos

### Tabular residual

`src/model/architecture.py`

- MLP simple sobre numéricas + embedding categórico + embedding de texto
- salida escalar de residual
- entrenamiento con `SmoothL1Loss`

### Secuencial residual

`src/model/ts_architecture.py`

- entrada secuencial: `price_yes_ffill`, `delta_price`, `observed_mask`
- GRU sobre 60 pasos
- rama estática con el mismo contexto snapshot-compatible del tabular
- fusión `GRU hidden + static branch -> MLP -> residual`

### Gradient Boosting

`src/model/gbdt_train.py`

- `HistGradientBoostingClassifier`
- features: numéricas snapshot-compatible + categoría one-hot + PCA del embedding de pregunta
- búsqueda corta de hiperparámetros en validación
- calibración automática `identity/platt/isotonic`
- hoy es el mejor modelo out-of-sample del repo

## Señales

El scorer expone estas columnas:

- `model_name`
- `price_yes`
- `p_yes_raw`
- `p_yes_calibrated`
- `ev_per_share`
- `expected_roi`
- `expected_roi_capped`
- `days_to_end`
- `signal`

Reglas por defecto:

- `BUY` si `ev_per_share >= 0.03` y `expected_roi_capped >= 0.10`
- `STRONG BUY` si `ev_per_share >= 0.07` y `expected_roi_capped >= 0.20`

Antes de etiquetar una señal, además se filtra por:

- `0.03 <= price_yes <= 0.85`
- `liquidity >= 1000`
- `volume_24h >= 100`
- `spread <= 0.10`

## Comandos

### Pipeline completo

```bash
python run.py
```

### Paso a paso

```bash
python -m src.features.pipeline --config config/config.yaml
python -m src.model.train --config config/config.yaml --only all
python -m src.scoring.scorer --config config/config.yaml --all-models
```

## Artefactos

### Dataset tabular

`data/processed/`

- `numerical_features.npy`
- `category_ids.npy`
- `text_embeddings.npy`
- `labels.npy`
- `targets.npy`
- `market_ids.npy`
- `snapshot_times.npy`
- `end_dates.npy`
- `days_to_end.npy`
- `snapshot_prices.npy`
- `metadata.json`
- `pipeline/`

### Dataset secuencial

`data/processed_ts/`

- `sequences.npy`
- `sequence_lengths.npy`
- `static_numerical.npy`
- `category_ids.npy`
- `text_embeddings.npy`
- `labels.npy`
- `targets.npy`
- `market_ids.npy`
- `snapshot_times.npy`
- `end_dates.npy`
- `days_to_end.npy`
- `snapshot_prices.npy`
- `metadata.json`

### Modelos

`data/models/`

- `market_value_baseline/`
- `price_sequence_gru/`
- `hist_gradient_boosting/`
- `v1/`
- `registry/`

`data/models/registry/`

- `model_comparison.json`
- `primary_model.json`
- `live_scores.csv`
- `live_scoring_summary.json`

## Validación esperada

Cada corrida de entrenamiento debe dejar:

- `Brier`, `log_loss`, `ROC-AUC`, `PR-AUC`, `ECE`
- comparación explícita contra `price_yes`
- métricas de EV y `Top-K`
- buckets `short_1_3d`, `medium_4_14d`, `long_15plus`
- selección automática del modelo primario

Estado actual esperado tras un rerun limpio:

- `hist_gradient_boosting` queda como primario
- supera a `price_yes` en `Brier` y `log_loss` sobre el split actual
- además lidera `Top-K avg realized pnl`

Si el modelo no supera al baseline de mercado en calibración o utilidad económica, no debe considerarse listo para producción.

## De Dónde Partir

Si solo quieres correr todo desde cero:

```bash
python run.py --force
```

Si quieres inspeccionar el flujo paso a paso, el orden correcto es:

1. `00_playground.ipynb`
2. `01_data_exploration.ipynb`
3. `02_feature_engineering.ipynb`
4. `03_processed_dataset_eda.ipynb`
5. `03_1_ts_dataset_validation.ipynb`
6. `04_model_training.ipynb`
7. `04_1_ts_model_training.ipynb`
8. `05_live_scoring.ipynb`
9. `05_1_ts_live_scoring.ipynb`
10. `05_2_model_comparison.ipynb`

## Notebooks

Los notebooks se mantienen como apoyo exploratorio y de validación visual, pero ya están alineados al pipeline actual.

- `00_playground.ipynb`: sanity check de `data/raw/`
- `01_data_exploration.ipynb`: calidad de datos, cobertura y compatibilidad histórica
- `02_feature_engineering.ipynb`: snapshot anti-leakage y validación de features
- `03_processed_dataset_eda.ipynb`: inspección del dataset tabular ya procesado
- `03_1_ts_dataset_validation.ipynb`: inspección del dataset secuencial
- `04_model_training.ipynb`: entrenamiento y revisión del tabular MLP
- `04_1_ts_model_training.ipynb`: entrenamiento y revisión del GRU híbrido
- `05_live_scoring.ipynb`: scoring vivo del tabular actual
- `05_1_ts_live_scoring.ipynb`: scoring vivo del GRU actual
- `05_2_model_comparison.ipynb`: comparación final entre MLP, GRU y GBDT

La validación principal y los artefactos de producción viven en scripts y en `data/models/registry/`.
