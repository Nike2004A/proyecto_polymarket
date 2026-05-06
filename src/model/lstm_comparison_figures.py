"""Generate report-ready figures for LSTM vs HistBoost comparison."""

from __future__ import annotations

import argparse
import json
import os
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

import numpy as np
from matplotlib import pyplot as plt


MODEL_DIRS = {
    "LSTM": "price_sequence_lstm",
    "HistBoost": "hist_gradient_boosting",
    "CatBoost": "catboost_residual",
}
MODEL_COLORS = {
    "LSTM": "#2A9D8F",
    "HistBoost": "#264653",
    "CatBoost": "#E76F51",
    "Mercado": "#8D99AE",
}
TEXT_COLOR = "#1D2630"
GRID_COLOR = "#D6DAE0"
BG_COLOR = "#F7F5F0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", default=str(ROOT / "data" / "models"))
    parser.add_argument("--ts-data-dir", default=str(ROOT / "data" / "processed_ts"))
    parser.add_argument("--out-dir", default=str(ROOT / "figures" / "lstm_comparison"))
    return parser.parse_args()


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def setup_style() -> None:
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
            "grid.color": GRID_COLOR,
            "grid.alpha": 0.45,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
        }
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def load_metrics(models_root: Path) -> dict[str, dict]:
    metrics = {}
    for label, model_dir in MODEL_DIRS.items():
        payload = load_json(models_root / model_dir / "test_metrics.json")
        if payload and payload.get("test"):
            metrics[label] = payload["test"]
    return metrics


def plot_scorecard(metrics: dict[str, dict], out_dir: Path) -> None:
    if not metrics:
        return

    names = list(metrics)
    metric_specs = [
        ("Brier", "brier", False),
        ("Log-loss", "log_loss", False),
        ("ECE", "ece", False),
        ("ROC-AUC", "roc_auc", True),
        ("PR-AUC", "pr_auc", True),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), gridspec_kw={"width_ratios": [1.25, 0.95]})
    ax_prob, ax_ev = axes

    x = np.arange(len(metric_specs))
    width = 0.18
    offsets = np.linspace(-width * len(names) / 2, width * len(names) / 2, len(names))
    for offset, name in zip(offsets, names):
        values = [metrics[name]["calibrated_metrics"][key] for _, key, _ in metric_specs]
        ax_prob.bar(x + offset, values, width=width, label=name, color=MODEL_COLORS.get(name, "#555"))

    market_values = [next(iter(metrics.values()))["market_baseline_metrics"][key] for _, key, _ in metric_specs]
    ax_prob.scatter(x, market_values, label="Mercado", color=MODEL_COLORS["Mercado"], marker="D", zorder=3)
    ax_prob.set_xticks(x)
    ax_prob.set_xticklabels([label for label, _, _ in metric_specs], rotation=20, ha="right")
    ax_prob.set_title("Calidad probabilística en test")
    ax_prob.grid(axis="y")
    ax_prob.legend(frameon=False)

    ev_labels = ["Top-K realized PnL", "Top-K hit rate"]
    ev_keys = ["top_k_avg_realized_pnl", "top_k_hit_rate"]
    x_ev = np.arange(len(ev_labels))
    for offset, name in zip(offsets, names):
        values = [metrics[name]["ev_metrics"][key] for key in ev_keys]
        ax_ev.bar(x_ev + offset, values, width=width, label=name, color=MODEL_COLORS.get(name, "#555"))
    ax_ev.axhline(0.0, color="#555", linewidth=1)
    ax_ev.set_xticks(x_ev)
    ax_ev.set_xticklabels(ev_labels, rotation=15, ha="right")
    ax_ev.set_title("Utilidad económica Top-K")
    ax_ev.grid(axis="y")

    fig.suptitle("LSTM vs HistBoost vs mercado", fontsize=18, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, out_dir / "lstm_vs_histboost_scorecard.png")


def plot_training_curves(models_root: Path, out_dir: Path) -> None:
    history = load_json(models_root / "price_sequence_lstm" / "training_history.json")
    if not history:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    epochs = np.arange(1, len(history.get("train_loss", [])) + 1)
    if epochs.size:
        axes[0].plot(epochs, history["train_loss"], label="Train loss", color=MODEL_COLORS["LSTM"])
        axes[0].plot(epochs, history["val_loss"], label="Val loss", color=MODEL_COLORS["HistBoost"])
        axes[0].set_title("Curvas de pérdida LSTM seleccionado")
        axes[0].set_xlabel("Epoch")
        axes[0].legend(frameon=False)
        axes[0].grid(axis="y")

        axes[1].plot(epochs, history.get("val_brier", []), label="Val Brier", color=MODEL_COLORS["LSTM"])
        axes[1].plot(epochs, history.get("val_log_loss", []), label="Val log-loss", color=MODEL_COLORS["CatBoost"])
        axes[1].set_title("Validación probabilística")
        axes[1].set_xlabel("Epoch")
        axes[1].legend(frameon=False)
        axes[1].grid(axis="y")

    fig.tight_layout()
    save_figure(fig, out_dir / "lstm_training_curves.png")


def plot_horizon_metrics(metrics: dict[str, dict], out_dir: Path) -> None:
    if not metrics:
        return

    buckets = ["short_1_3d", "medium_4_14d", "long_15plus"]
    labels = ["1-3d", "4-14d", "15d+"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    width = 0.22
    x = np.arange(len(buckets))
    offsets = np.linspace(-width * len(metrics) / 2, width * len(metrics) / 2, len(metrics))

    for offset, (name, payload) in zip(offsets, metrics.items()):
        brier_values = [
            payload.get("by_horizon", {}).get(bucket, {}).get("probability_metrics", {}).get("brier", np.nan)
            for bucket in buckets
        ]
        pnl_values = [
            payload.get("by_horizon", {}).get(bucket, {}).get("ev_metrics", {}).get("top_k_avg_realized_pnl", np.nan)
            for bucket in buckets
        ]
        axes[0].bar(x + offset, brier_values, width=width, label=name, color=MODEL_COLORS.get(name, "#555"))
        axes[1].bar(x + offset, pnl_values, width=width, label=name, color=MODEL_COLORS.get(name, "#555"))

    axes[0].set_title("Brier por horizonte")
    axes[1].set_title("Top-K realized PnL por horizonte")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid(axis="y")
    axes[0].legend(frameon=False)
    axes[1].axhline(0.0, color="#555", linewidth=1)

    fig.tight_layout()
    save_figure(fig, out_dir / "lstm_vs_histboost_by_horizon.png")


def plot_sequence_example(ts_data_dir: Path, out_dir: Path) -> None:
    sequences_path = ts_data_dir / "sequences.npy"
    labels_path = ts_data_dir / "labels.npy"
    if not sequences_path.exists():
        return

    sequences = np.load(sequences_path)
    labels = np.load(labels_path) if labels_path.exists() else np.zeros(len(sequences))
    idx = int(np.flatnonzero(labels == 1)[0]) if np.any(labels == 1) else 0
    sequence = sequences[idx]
    steps = np.arange(1, sequence.shape[0] + 1)

    fig, ax_price = plt.subplots(figsize=(12, 5.2))
    ax_price.plot(steps, sequence[:, 0], color=MODEL_COLORS["LSTM"], linewidth=2.2, label="price_yes_ffill")
    ax_price.bar(steps, sequence[:, 2] * 0.05, bottom=0.0, color="#ADB5BD", alpha=0.5, label="observed_mask")
    ax_price.set_title("Ejemplo de secuencia que recibe el LSTM")
    ax_price.set_xlabel("Paso de 12 horas dentro de ventana de 30 días")
    ax_price.set_ylabel("Precio YES")
    ax_price.set_ylim(0, 1)
    ax_price.grid(axis="y")
    ax_price.legend(frameon=False)

    save_figure(fig, out_dir / "lstm_sequence_example.png")


def save_metrics_table(metrics: dict[str, dict], out_dir: Path) -> None:
    if not metrics:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        [
            "model",
            "brier",
            "log_loss",
            "ece",
            "roc_auc",
            "pr_auc",
            "top_k_avg_realized_pnl",
            "top_k_hit_rate",
        ]
    ]
    for name, payload in metrics.items():
        prob = payload["calibrated_metrics"]
        ev = payload["ev_metrics"]
        rows.append([
            name,
            prob["brier"],
            prob["log_loss"],
            prob["ece"],
            prob["roc_auc"],
            prob["pr_auc"],
            ev["top_k_avg_realized_pnl"],
            ev["top_k_hit_rate"],
        ])
    rows.append([
        "Mercado",
        next(iter(metrics.values()))["market_baseline_metrics"]["brier"],
        next(iter(metrics.values()))["market_baseline_metrics"]["log_loss"],
        next(iter(metrics.values()))["market_baseline_metrics"]["ece"],
        next(iter(metrics.values()))["market_baseline_metrics"]["roc_auc"],
        next(iter(metrics.values()))["market_baseline_metrics"]["pr_auc"],
        "",
        "",
    ])
    with open(out_dir / "metrics_table.csv", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(",".join(str(value) for value in row) + "\n")


def main() -> None:
    args = parse_args()
    setup_style()
    models_root = Path(args.models_root)
    ts_data_dir = Path(args.ts_data_dir)
    out_dir = Path(args.out_dir)

    metrics = load_metrics(models_root)
    save_metrics_table(metrics, out_dir)
    plot_scorecard(metrics, out_dir)
    plot_training_curves(models_root, out_dir)
    plot_horizon_metrics(metrics, out_dir)
    plot_sequence_example(ts_data_dir, out_dir)
    print(f"Figuras exportadas en {out_dir}")


if __name__ == "__main__":
    main()
