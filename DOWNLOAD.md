# Guía de Ejecución — Polymarket ML Pipeline

Este documento describe los comandos exactos para ejecutar cada fase del pipeline,
en orden, con instrucciones de reanudación si algo se interrumpe.

---

## 0. Entorno

```bash
source .venv/bin/activate
```

Todos los comandos asumen que estás en la raíz del proyecto.

---

## 1. Descarga de datos (`data/raw/`)

### Descarga completa (primera vez)

```bash
python -m src.data.fetcher --mode full
```

Descarga en orden:
1. Tags/categorías → `data/raw/tags.json`
2. Mercados activos (~3k) → `data/raw/active_markets.json`
3. Mercados resueltos (~7k) → `data/raw/resolved_markets.json`
4. Order books para activos → `data/raw/order_books.json`
5. Price histories para activos + resueltos → `data/raw/price_histories.json`

El paso 5 es el más largo (~3 horas para 20k mercados a 1.8 it/s).
La descarga de histories es **incremental**: si se interrumpe, retomar con:

```bash
python -m src.data.fetcher --mode histories
```

Esto carga lo que ya está en disco y salta los market_ids ya descargados.
Guarda un checkpoint cada 200 mercados para no perder progreso.

### Verificar resultado

```bash
python -c "
import json
h = json.load(open('data/raw/fetch_metadata.json'))
print(h)
"
```

Esperar ver `resolved_histories_coverage_pct` > 80%.

### Opciones útiles

```bash
# Solo actualizar mercados activos (para scoring periódico)
python -m src.data.fetcher --mode active

# Solo mercados resueltos (para ampliar training data)
python -m src.data.fetcher --mode resolved

# Solo completar histories faltantes
python -m src.data.fetcher --mode histories

# Cambiar límites en una corrida
python -m src.data.fetcher --mode full --max-active 3000 --max-resolved 7000
```

---

## 2. EDA de datos crudos (`notebooks/01_data_exploration.ipynb`)

Se puede correr mientras descarga (con los datos parciales que hay en disco).

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

O ejecutar directamente:

```bash
jupyter nbconvert --to notebook --execute notebooks/01_data_exploration.ipynb --output notebooks/01_data_exploration.ipynb
```

Genera figuras en `figures/01_*.png`. Lo que muestra:
- Cobertura de price histories (activos vs resueltos)
- Distribución de longitud de series
- Gaps temporales
- Balance de labels Yes/No
- Features numéricas clave

---

## 3. Feature engineering → datasets procesados (`data/processed/`)

Requiere que `data/raw/resolved_markets.json` y `data/raw/price_histories.json` existan.

```bash
python -m src.features.pipeline
```

Genera en `data/processed/`:

| Archivo | Shape | Contenido |
|---|---|---|
| `numerical_features.npy` | `(N, 23)` | Features numéricas normalizadas |
| `labels.npy` | `(N,)` | 0/1 — ¿resolvió Yes con ≥5% retorno? |
| `text_embeddings.npy` | `(N, 384)` | MiniLM sobre la pregunta del mercado |
| `category_ids.npy` | `(N,)` | Categoría codificada (entero) |
| `end_dates.npy` | `(N,)` | Timestamp para split temporal |
| `pipeline/` | — | Scaler y CategoryEncoder serializados |

Si no tenés `sentence-transformers` instalado:

```bash
python -m src.features.pipeline --dummy-text
```

### Cambiar snapshot offset (anti-leakage)

Por default toma el precio 7 días antes de resolución.
Para experimentar con otros offsets:

```bash
python -m src.features.pipeline --snapshot-offset 14
python -m src.features.pipeline --snapshot-offset 30
```

---

## 4. Entrenamiento (`data/models/`)

> Requiere `data/processed/` completo del paso 3.

```bash
python -m src.model.train --data-dir data/processed --epochs 50 --batch-size 64
```

Opciones:

```bash
--epochs 50          # épocas de entrenamiento
--batch-size 64      # tamaño de batch
--lr 0.001           # learning rate
--data-dir ...       # directorio con los .npy
--save-dir ...       # dónde guardar el modelo (default: data/models)
```

El split es **temporal**: los mercados más viejos van a train, los más recientes a val.
Esto simula correctamente la predicción sobre mercados futuros.

---

## 5. Scoring en vivo

> Requiere modelo entrenado en `data/models/`.

```bash
python -m src.scoring.scorer --top 20
```

Opera sobre `data/raw/active_markets.json`. Produce señales BUY/STRONG_BUY
según los umbrales en `config/config.yaml` (`scoring.buy_threshold`, `scoring.strong_buy_threshold`).

---

## Flujo completo (resumen)

```
1. python -m src.data.fetcher --mode full        # ~90 min
2. [reanudar si se interrumpe]
   python -m src.data.fetcher --mode histories
3. jupyter notebook notebooks/01_data_exploration.ipynb
4. python -m src.features.pipeline               # ~5 min
5. python -m src.model.train                     # ~10 min
6. python -m src.scoring.scorer --top 20
```
