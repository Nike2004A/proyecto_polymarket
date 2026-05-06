from __future__ import annotations

import argparse
import json
import os
import textwrap
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
cache_root = ROOT / ".cache"
mpl_root = ROOT / ".mplconfig"
fontconfig_root = cache_root / "fontconfig"
for path in (cache_root, mpl_root, fontconfig_root):
    path.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(mpl_root))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.inspection import permutation_importance

from ..config import load_config
from .dataset import PolymarketDataset
from .splits import build_dataset_split

MODEL_COLOR = "#0F4C81"
MARKET_COLOR = "#A7B1C2"
POSITIVE_COLOR = "#1F9D8B"
WARN_COLOR = "#F4A261"
NEGATIVE_COLOR = "#D1495B"
TEXT_COLOR = "#14213D"
GRID_COLOR = "#D7DCE2"
BG_COLOR = "#F6F3EE"


@dataclass
class GBDTArtifacts:
    cfg: dict
    run_config: dict
    test_metrics: dict
    training_history: dict
    live_df: pd.DataFrame
    live_summary: dict
    model: object
    test_df: pd.DataFrame
    feature_names_num: list[str]
    cat_feature_dim: int
    text_pca_dim: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate presentation-ready GBDT figures.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "config" / "config.yaml"),
        help="Path to config/config.yaml",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(ROOT / "figures" / "presentation"),
        help="Directory to store exported PNG figures",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Primary K to highlight in scorecards and economic plots",
    )
    return parser.parse_args()


def setup_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": BG_COLOR,
            "axes.facecolor": BG_COLOR,
            "savefig.facecolor": BG_COLOR,
            "axes.edgecolor": GRID_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "text.color": TEXT_COLOR,
            "axes.titleweight": "bold",
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "axes.titlepad": 16,
            "font.size": 11,
            "grid.color": GRID_COLOR,
            "grid.alpha": 0.45,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
        }
    )


def load_artifacts(config_path: str) -> GBDTArtifacts:
    cfg = load_config(config_path)
    processed_dir = ROOT / cfg["data"]["processed_dir"]
    save_dir = ROOT / "data" / "models" / "hist_gradient_boosting"
    registry_dir = ROOT / "data" / "models" / "registry"

    run_config = json.loads((save_dir / "run_config.json").read_text())
    test_metrics = json.loads((save_dir / "test_metrics.json").read_text())
    training_history = json.loads((save_dir / "training_history.json").read_text())
    live_summary = json.loads((registry_dir / "live_scoring_summary.json").read_text()).get(
        "hist_gradient_boosting", {}
    )
    live_df = pd.read_csv(registry_dir / "live_scores.csv")
    live_df = live_df[live_df["model_name"] == "hist_gradient_boosting"].copy()

    dataset = PolymarketDataset.from_numpy_dir(str(processed_dir))
    split_cfg = run_config["split"]
    split = build_dataset_split(
        n_samples=len(dataset),
        labels=dataset.labels.numpy(),
        timestamps=dataset.timestamps,
        groups=dataset.groups,
        val_split=float(split_cfg["val_split"]),
        test_split=float(split_cfg["test_split"]),
        strategy=split_cfg["strategy"],
        seed=int(split_cfg["seed"]),
    )
    test_idx = np.asarray(split.test_indices, dtype=np.int64)

    model = joblib.load(save_dir / "gbdt_model.pkl")
    cat_enc = joblib.load(save_dir / "category_encoder.pkl")
    text_pca = joblib.load(save_dir / "text_pca.pkl")

    numerical = dataset.numerical.numpy()
    categories = dataset.categories.numpy()
    text_embeddings = dataset.text_emb.numpy()
    labels = dataset.labels.numpy()
    market_probs = np.asarray(dataset.snapshot_prices)

    cat_features = cat_enc.transform(categories[test_idx].reshape(-1, 1))
    text_features = text_pca.transform(text_embeddings[test_idx])
    X_test = np.concatenate([numerical[test_idx], cat_features, text_features], axis=1).astype(np.float32)
    y_true = labels[test_idx]
    y_model = model.predict_proba(X_test)[:, 1]
    y_market = market_probs[test_idx]
    edge = y_model - y_market
    realized_pnl = y_true - y_market

    feature_names_num = json.loads((processed_dir / "metadata.json").read_text())["tabular_feature_names"]
    test_df = pd.DataFrame(
        {
            "y_true": y_true,
            "model_prob": y_model,
            "market_prob": y_market,
            "edge": edge,
            "realized_pnl": realized_pnl,
            "days_to_end": np.asarray(dataset.days_to_end)[test_idx],
        }
    )

    return GBDTArtifacts(
        cfg=cfg,
        run_config=run_config,
        test_metrics=test_metrics,
        training_history=training_history,
        live_df=live_df,
        live_summary=live_summary,
        model=model,
        test_df=test_df,
        feature_names_num=feature_names_num,
        cat_feature_dim=cat_features.shape[1],
        text_pca_dim=text_features.shape[1],
    )


def calibration_by_quantile(scores: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"score": scores, "label": labels})
    df["bucket"] = pd.qcut(df["score"], q=n_bins, duplicates="drop")
    return (
        df.groupby("bucket", observed=True)
        .agg(mean_score=("score", "mean"), win_rate=("label", "mean"), count=("label", "size"))
        .reset_index()
    )


def save_figure(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def add_figure_header(
    fig: plt.Figure,
    title: str,
    subtitle: str,
    *,
    top: float = 0.78,
    bottom: float = 0.10,
    left: float = 0.06,
    right: float = 0.98,
    subtitle_width: int = 125,
) -> None:
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
    fig.suptitle(
        title,
        x=left,
        y=0.985,
        ha="left",
        va="top",
        fontsize=20,
        fontweight="bold",
    )
    fig.text(
        left,
        0.938,
        textwrap.fill(subtitle, width=subtitle_width),
        ha="left",
        va="top",
        fontsize=11,
        color="#58677A",
    )


def plot_scorecard(artifacts: GBDTArtifacts, out_dir: Path, top_k: int) -> None:
    test = artifacts.test_metrics["test"]
    raw = test["raw_metrics"]
    market = test["market_baseline_metrics"]
    ev = test["ev_metrics"]

    lower_better = pd.DataFrame(
        {
            "metric": ["Brier", "Log-loss", "ECE"],
            "GBDT": [raw["brier"], raw["log_loss"], raw["ece"]],
            "Mercado": [market["brier"], market["log_loss"], market["ece"]],
        }
    )
    higher_better = pd.DataFrame(
        {
            "metric": ["ROC-AUC", "PR-AUC"],
            "GBDT": [raw["roc_auc"], raw["pr_auc"]],
            "Mercado": [market["roc_auc"], market["pr_auc"]],
        }
    )

    fig = plt.figure(figsize=(15.8, 7.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 0.95, 1.1], wspace=0.32)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_mid = fig.add_subplot(gs[0, 1])
    ax_right = fig.add_subplot(gs[0, 2])

    width = 0.35
    x = np.arange(len(lower_better))
    ax_left.bar(x - width / 2, lower_better["GBDT"], width, color=MODEL_COLOR, label="GBDT")
    ax_left.bar(x + width / 2, lower_better["Mercado"], width, color=MARKET_COLOR, label="Mercado")
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(lower_better["metric"])
    ax_left.set_title("Calidad probabilística")
    ax_left.set_ylabel("Menor es mejor")
    ax_left.set_ylim(0, lower_better[["GBDT", "Mercado"]].to_numpy().max() * 1.18)
    ax_left.legend(frameon=False, loc="upper right")
    for idx, row in lower_better.iterrows():
        delta = row["Mercado"] - row["GBDT"]
        ax_left.text(
            idx,
            max(row["GBDT"], row["Mercado"]) * 1.08,
            f"{delta:+.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color=POSITIVE_COLOR if delta > 0 else NEGATIVE_COLOR,
            bbox={"facecolor": BG_COLOR, "edgecolor": "none", "pad": 0.2, "alpha": 0.95},
        )

    x = np.arange(len(higher_better))
    ax_mid.bar(x - width / 2, higher_better["GBDT"], width, color=MODEL_COLOR, label="GBDT")
    ax_mid.bar(x + width / 2, higher_better["Mercado"], width, color=MARKET_COLOR, label="Mercado")
    ax_mid.set_xticks(x)
    ax_mid.set_xticklabels(higher_better["metric"])
    ax_mid.set_title("Capacidad de ranking")
    ax_mid.set_ylabel("Mayor es mejor")
    rank_values = higher_better[["GBDT", "Mercado"]].to_numpy()
    ax_mid.set_ylim(rank_values.min() - 0.0008, rank_values.max() + 0.0003)
    for idx, row in higher_better.iterrows():
        delta = row["GBDT"] - row["Mercado"]
        ax_mid.text(
            idx,
            max(row["GBDT"], row["Mercado"]) + 0.00005,
            f"{delta:+.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color=POSITIVE_COLOR if delta > 0 else NEGATIVE_COLOR,
            bbox={"facecolor": BG_COLOR, "edgecolor": "none", "pad": 0.2, "alpha": 0.95},
        )

    ax_right.axis("off")
    ax_right.set_title("KPIs para la historia", loc="left")
    cards = [
        (f"Top-{top_k} hit rate", f"{ev['top_k_hit_rate']:.0%}", POSITIVE_COLOR),
        (f"Top-{top_k} avg realized PnL", f"{ev['top_k_avg_realized_pnl']:+.3f}", POSITIVE_COLOR),
        ("Mercados live accionables", f"{int((artifacts.live_df['signal'] != 'HOLD').sum())}/{len(artifacts.live_df)}", WARN_COLOR),
        ("Brier vs mercado", f"{((market['brier'] - raw['brier']) / market['brier']):+.1%}", POSITIVE_COLOR),
    ]
    card_positions = [(0.00, 0.58), (0.52, 0.58), (0.00, 0.18), (0.52, 0.18)]
    for (title, value, color), (x0, y0) in zip(cards, card_positions):
        patch = FancyBboxPatch(
            (x0, y0),
            0.45,
            0.25,
            boxstyle="round,pad=0.012,rounding_size=0.03",
            transform=ax_right.transAxes,
            facecolor="white",
            edgecolor="none",
            alpha=0.95,
        )
        ax_right.add_patch(patch)
        ax_right.text(
            x0 + 0.04,
            y0 + 0.16,
            textwrap.fill(title, width=19),
            transform=ax_right.transAxes,
            fontsize=9.5,
            color="#58677A",
            va="center",
            linespacing=1.15,
        )
        ax_right.text(
            x0 + 0.04,
            y0 + 0.05,
            value,
            transform=ax_right.transAxes,
            fontsize=20,
            fontweight="bold",
            color=color,
        )

    add_figure_header(
        fig,
        "GBDT: scorecard de presentación",
        f"Test temporal hold-out • {len(artifacts.test_df):,} muestras • K destacado = {top_k}",
        top=0.78,
        bottom=0.10,
        subtitle_width=80,
    )
    save_figure(fig, out_dir / "gbdt_presentation_scorecard.png")


def plot_calibration_and_topk(artifacts: GBDTArtifacts, out_dir: Path) -> None:
    test_df = artifacts.test_df.sort_values("edge", ascending=False).reset_index(drop=True)
    calib_model = calibration_by_quantile(test_df["model_prob"].to_numpy(), test_df["y_true"].to_numpy())
    calib_market = calibration_by_quantile(test_df["market_prob"].to_numpy(), test_df["y_true"].to_numpy())

    k_grid = np.array([5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500])
    rows = []
    for k in k_grid:
        subset = test_df.head(k)
        rows.append(
            {
                "k": k,
                "predicted_edge": subset["edge"].mean(),
                "realized_pnl": subset["realized_pnl"].mean(),
                "hit_rate": subset["y_true"].mean(),
            }
        )
    topk_df = pd.DataFrame(rows)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(15.8, 7.4), gridspec_kw={"wspace": 0.24})

    ax_left.plot([0, 1], [0, 1], linestyle="--", color="#B8BDC7", lw=1.3, label="calibración perfecta")
    ax_left.plot(calib_market["mean_score"], calib_market["win_rate"], marker="o", color=MARKET_COLOR, lw=2.2, label="Mercado")
    ax_left.plot(calib_model["mean_score"], calib_model["win_rate"], marker="o", color=MODEL_COLOR, lw=2.6, label="GBDT")
    ax_left.scatter(calib_model["mean_score"], calib_model["win_rate"], s=calib_model["count"] / 8, color=MODEL_COLOR, alpha=0.22)
    ax_left.set_title("Calibración por deciles de probabilidad")
    ax_left.set_xlabel("Probabilidad media del bin")
    ax_left.set_ylabel("Frecuencia real de YES")
    ax_left.legend(frameon=False, loc="upper left")
    ax_left.set_xlim(0, 1)
    ax_left.set_ylim(0, 1)

    ax_right.plot(topk_df["k"], topk_df["predicted_edge"], color=MODEL_COLOR, lw=2.8, marker="o", label="EV predicho promedio")
    ax_right.plot(topk_df["k"], topk_df["realized_pnl"], color=POSITIVE_COLOR, lw=2.8, marker="o", label="PnL realizado promedio")
    ax_right.axhline(0, color="#7A869A", lw=1.2, linestyle="--")
    ax_right.set_title("Monetización de la cola alta")
    ax_right.set_xlabel("Top-K ordenado por edge del modelo")
    ax_right.set_ylabel("Retorno promedio por share")
    ax_right.legend(frameon=False, loc="upper right")
    ax_right2 = ax_right.twinx()
    ax_right2.plot(topk_df["k"], topk_df["hit_rate"], color=WARN_COLOR, lw=2.0, linestyle=":", marker="s", label="Hit rate")
    ax_right2.set_ylabel("Hit rate", color=WARN_COLOR)
    ax_right2.tick_params(axis="y", colors=WARN_COLOR)
    ax_right2.set_ylim(0, 1)
    ax_right.annotate(
        "K=20",
        xy=(20, topk_df.loc[topk_df["k"] == 20, "realized_pnl"].iloc[0]),
        xytext=(10, 16),
        textcoords="offset points",
        color=POSITIVE_COLOR,
        fontsize=10,
        bbox={"facecolor": BG_COLOR, "edgecolor": "none", "pad": 0.2, "alpha": 0.95},
    )
    ax_right.annotate(
        "K=100",
        xy=(100, topk_df.loc[topk_df["k"] == 100, "realized_pnl"].iloc[0]),
        xytext=(14, 12),
        textcoords="offset points",
        color=POSITIVE_COLOR,
        fontsize=10,
        bbox={"facecolor": BG_COLOR, "edgecolor": "none", "pad": 0.2, "alpha": 0.95},
    )

    add_figure_header(
        fig,
        "GBDT: calibración y captura de valor",
        "La izquierda responde si las probabilidades son creíbles; la derecha responde si la cola con más edge monetiza de verdad.",
        top=0.78,
        bottom=0.12,
        right=0.97,
        subtitle_width=110,
    )
    save_figure(fig, out_dir / "gbdt_presentation_calibration_topk.png")


def plot_horizon_breakdown(artifacts: GBDTArtifacts, out_dir: Path) -> None:
    horizon_labels = {
        "short_1_3d": "corto\n1-3d",
        "medium_4_14d": "medio\n4-14d",
        "long_15plus": "largo\n15+d",
    }
    rows = []
    for horizon, payload in artifacts.test_metrics["test"]["by_horizon"].items():
        prob = payload["probability_metrics"]
        market = payload["market_baseline_metrics"]
        ev = payload["ev_metrics"]
        rows.append(
            {
                "Horizon": horizon_labels.get(horizon, horizon.replace("_", " ").replace("plus", "+")),
                "count": payload["count"],
                "brier_gbdt": prob["brier"],
                "brier_market": market["brier"],
                "logloss_gbdt": prob["log_loss"],
                "logloss_market": market["log_loss"],
                "topk_hit": ev["top_k_hit_rate"],
                "topk_pnl": ev["top_k_avg_realized_pnl"],
            }
        )
    horizon_df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(16.8, 7.4), gridspec_kw={"wspace": 0.30})
    width = 0.35
    x = np.arange(len(horizon_df))

    axes[0].bar(x - width / 2, horizon_df["brier_gbdt"], width, color=MODEL_COLOR, label="GBDT")
    axes[0].bar(x + width / 2, horizon_df["brier_market"], width, color=MARKET_COLOR, label="Mercado")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(horizon_df["Horizon"])
    axes[0].set_title("Brier por horizonte")
    axes[0].set_ylabel("Menor es mejor")
    axes[0].legend(frameon=False, loc="upper right")

    axes[1].bar(x - width / 2, horizon_df["logloss_gbdt"], width, color=MODEL_COLOR, label="GBDT")
    axes[1].bar(x + width / 2, horizon_df["logloss_market"], width, color=MARKET_COLOR, label="Mercado")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(horizon_df["Horizon"])
    axes[1].set_title("Log-loss por horizonte")
    axes[1].set_ylabel("Menor es mejor")

    bars = axes[2].bar(horizon_df["Horizon"], horizon_df["topk_pnl"], color=POSITIVE_COLOR, alpha=0.9)
    axes[2].set_title("Top-K por horizonte")
    axes[2].set_ylabel("PnL promedio por share")
    axes[2].axhline(0, color="#7A869A", lw=1.2, linestyle="--")
    axes[2].set_ylim(0, horizon_df["topk_pnl"].max() * 1.28)
    ax2 = axes[2].twinx()
    ax2.plot(horizon_df["Horizon"], horizon_df["topk_hit"], color=WARN_COLOR, marker="o", lw=2.0)
    ax2.set_ylabel("Hit rate", color=WARN_COLOR)
    ax2.tick_params(axis="y", colors=WARN_COLOR)
    ax2.set_ylim(0, 1)
    for bar, pnl, hit in zip(bars, horizon_df["topk_pnl"], horizon_df["topk_hit"]):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            pnl + horizon_df["topk_pnl"].max() * 0.04,
            f"{pnl:+.3f}\n{hit:.0%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    add_figure_header(
        fig,
        "GBDT: dónde gana según el horizonte",
        "La mejora probabilística es consistente en todo el hold-out, pero el payoff económico se concentra más en el tramo largo.",
        top=0.78,
        bottom=0.12,
        right=0.96,
        subtitle_width=110,
    )
    save_figure(fig, out_dir / "gbdt_presentation_horizons.png")


def plot_search_frontier(artifacts: GBDTArtifacts, out_dir: Path) -> None:
    rows = []
    selected = artifacts.run_config["model"]
    for item in artifacts.training_history.get("candidates", []):
        params = item["model_params"]
        val = item["validation"]
        short_label = f"lr={params['learning_rate']}\nd={params['max_depth']}"
        rows.append(
            {
                "label": f"lr={params['learning_rate']}, d={params['max_depth']}",
                "short_label": short_label,
                "learning_rate": params["learning_rate"],
                "max_depth": params["max_depth"],
                "max_iter": params["max_iter"],
                "brier": val["brier"],
                "log_loss": val["log_loss"],
                "topk_pnl": val["top_k_avg_realized_pnl"],
                "hit_rate": val["top_k_hit_rate"],
                "selected": (
                    params["learning_rate"] == selected["learning_rate"]
                    and params["max_depth"] == selected["max_depth"]
                    and params["max_iter"] == selected["max_iter"]
                    and params["min_samples_leaf"] == selected["min_samples_leaf"]
                    and params["l2_regularization"] == selected["l2_regularization"]
                ),
            }
        )
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(15.8, 7.4), gridspec_kw={"wspace": 0.30})

    sizes = 800 * df["hit_rate"]
    colors = np.where(df["selected"], MODEL_COLOR, "#A9B2C3")
    axes[0].scatter(df["brier"], df["topk_pnl"], s=sizes, c=colors, alpha=0.9, edgecolors="white", linewidth=1.2)
    annotation_offsets = {
        "lr=0.03, d=4": (8, 10),
        "lr=0.03, d=6": (8, -18),
        "lr=0.05, d=6": (8, 10),
        "lr=0.05, d=8": (8, 8),
        "lr=0.07, d=6": (10, 6),
        "lr=0.1, d=4": (8, -6),
    }
    for _, row in df.iterrows():
        dx, dy = annotation_offsets.get(row["label"], (8, 8))
        axes[0].annotate(
            row["label"],
            xy=(row["brier"], row["topk_pnl"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=9,
            color=TEXT_COLOR,
            bbox={"facecolor": BG_COLOR, "edgecolor": "none", "pad": 0.2, "alpha": 0.95},
        )
    axes[0].set_title("Frontera validación: riesgo vs payoff")
    axes[0].set_xlabel("Brier en validación")
    axes[0].set_ylabel("Top-K realized PnL")

    axes[1].bar(df["short_label"], df["hit_rate"], color=colors, alpha=0.95)
    axes[1].set_title("Hit rate del Top-K por candidato")
    axes[1].set_ylabel("Hit rate")
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis="x", rotation=0)
    for idx, value in enumerate(df["hit_rate"]):
        axes[1].text(idx, value + 0.03, f"{value:.0%}", ha="center", fontsize=9)

    add_figure_header(
        fig,
        "GBDT: selección de hiperparámetros",
        "El modelo elegido no es el más conservador en Brier; se eligió por mejor payoff y mayor hit rate en la cola accionable.",
        top=0.78,
        bottom=0.14,
        subtitle_width=110,
    )
    save_figure(fig, out_dir / "gbdt_presentation_search.png")


def plot_feature_drivers(artifacts: GBDTArtifacts, out_dir: Path) -> None:
    test_df = artifacts.test_df
    dataset = PolymarketDataset.from_numpy_dir(str(ROOT / artifacts.cfg["data"]["processed_dir"]))
    split_cfg = artifacts.run_config["split"]
    split = build_dataset_split(
        n_samples=len(dataset),
        labels=dataset.labels.numpy(),
        timestamps=dataset.timestamps,
        groups=dataset.groups,
        val_split=float(split_cfg["val_split"]),
        test_split=float(split_cfg["test_split"]),
        strategy=split_cfg["strategy"],
        seed=int(split_cfg["seed"]),
    )
    test_idx = np.asarray(split.test_indices, dtype=np.int64)
    cat_enc = joblib.load(ROOT / "data" / "models" / "hist_gradient_boosting" / "category_encoder.pkl")
    text_pca = joblib.load(ROOT / "data" / "models" / "hist_gradient_boosting" / "text_pca.pkl")
    categories = dataset.categories.numpy()
    text_embeddings = dataset.text_emb.numpy()
    numerical = dataset.numerical.numpy()
    cat_features = cat_enc.transform(categories[test_idx].reshape(-1, 1))
    text_features = text_pca.transform(text_embeddings[test_idx])
    X_test = np.concatenate([numerical[test_idx], cat_features, text_features], axis=1).astype(np.float32)

    perm = permutation_importance(
        artifacts.model,
        X_test,
        test_df["y_true"].to_numpy(),
        scoring="roc_auc",
        n_repeats=5,
        random_state=int(split_cfg["seed"]),
        n_jobs=1,
    )
    importances = perm.importances_mean
    stds = perm.importances_std

    n_num = len(artifacts.feature_names_num)
    n_cat = artifacts.cat_feature_dim
    n_pca = artifacts.text_pca_dim
    grouped = pd.DataFrame(
        {
            "feature": artifacts.feature_names_num + ["categoría (OHE)", f"texto (PCA {n_pca}D)"],
            "importance": list(importances[:n_num]) + [importances[n_num : n_num + n_cat].sum(), importances[n_num + n_cat :].sum()],
            "std": list(stds[:n_num])
            + [np.sqrt((stds[n_num : n_num + n_cat] ** 2).sum()), np.sqrt((stds[n_num + n_cat :] ** 2).sum())],
        }
    ).sort_values("importance", ascending=False)
    top = grouped.head(12).iloc[::-1]

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    colors = [WARN_COLOR if name in {"snapshot_price_yes", "neg_risk"} else MODEL_COLOR for name in top["feature"]]
    ax.barh(top["feature"], top["importance"], xerr=top["std"], color=colors, edgecolor="white", ecolor="#455A64")
    ax.set_title("Drivers del GBDT por importancia de permutación")
    ax.set_xlabel("Caída media de ROC-AUC al permutar la feature")
    ax.axvline(0, color="#7A869A", lw=1.2)
    ax.text(
        0.98,
        0.02,
        "snapshot_price_yes domina; el resto añade señal incremental fina.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        color="#58677A",
    )

    fig.suptitle("GBDT: qué está moviendo la decisión", x=0.06, y=0.98, ha="left", fontsize=18, fontweight="bold")
    fig.text(0.06, 0.92, "Importancia estimada sobre el test hold-out con permutation importance (ROC-AUC).", fontsize=11, color="#58677A")
    save_figure(fig, out_dir / "gbdt_presentation_feature_drivers.png")


def shorten(text: str, width: int = 52) -> str:
    return textwrap.shorten(" ".join(text.split()), width=width, placeholder="…")


def plot_live_signals(artifacts: GBDTArtifacts, out_dir: Path) -> None:
    df = artifacts.live_df.copy()
    actionable = df[df["signal"] != "HOLD"].copy().sort_values("ev_per_share", ascending=False).head(12)

    fig, axes = plt.subplots(1, 2, figsize=(16.0, 6.2), gridspec_kw={"width_ratios": [0.9, 1.5], "wspace": 0.26})

    signal_counts = df["signal"].value_counts().reindex(["HOLD", "BUY", "STRONG BUY"], fill_value=0)
    colors = [MARKET_COLOR, WARN_COLOR, POSITIVE_COLOR]
    bars = axes[0].bar(signal_counts.index, signal_counts.values, color=colors, width=0.58)
    axes[0].set_title("Funnel de señales live")
    axes[0].set_ylabel("Mercados")
    axes[0].set_ylim(0, signal_counts.max() * 1.18)
    for bar, value in zip(bars, signal_counts.values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value + signal_counts.max() * 0.03,
            f"{value}\n({value / len(df):.1%})",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    if not actionable.empty:
        actionable["label"] = actionable["question"].map(lambda x: shorten(x, width=55))
        palette = {"BUY": WARN_COLOR, "STRONG BUY": POSITIVE_COLOR}
        axes[1].barh(actionable["label"][::-1], actionable["ev_per_share"][::-1], color=[palette[s] for s in actionable["signal"][::-1]])
        axes[1].set_title("Top oportunidades accionables")
        axes[1].set_xlabel("EV por share")
        axes[1].axvline(0, color="#7A869A", lw=1.2)
        for idx, row in actionable.iloc[::-1].reset_index(drop=True).iterrows():
            axes[1].text(
                row["ev_per_share"] + 0.006,
                idx,
                f"{row['signal']} • {row['days_to_end']:.0f}d",
                va="center",
                fontsize=9,
                color="#58677A",
            )

    fig.suptitle("GBDT: señal live para slides", x=0.06, y=0.98, ha="left", fontsize=18, fontweight="bold")
    fig.text(
        0.06,
        0.92,
        "El modelo filtra fuerte: pocas oportunidades accionables, pero con una cola claramente separada por edge esperado.",
        fontsize=11,
        color="#58677A",
    )
    save_figure(fig, out_dir / "gbdt_presentation_live_signals.png")


def main() -> None:
    args = parse_args()
    setup_style()
    artifacts = load_artifacts(args.config)
    out_dir = Path(args.out_dir)

    plot_scorecard(artifacts, out_dir, top_k=args.top_k)
    plot_calibration_and_topk(artifacts, out_dir)
    plot_horizon_breakdown(artifacts, out_dir)
    plot_search_frontier(artifacts, out_dir)
    plot_feature_drivers(artifacts, out_dir)
    plot_live_signals(artifacts, out_dir)

    print("Generated presentation figures:")
    for path in sorted(out_dir.glob("gbdt_presentation_*.png")):
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
