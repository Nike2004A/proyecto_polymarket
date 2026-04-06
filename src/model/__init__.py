"""API pública ligera para los módulos de modelado."""

__all__ = [
    "MarketValueNet",
    "PolymarketDataset",
    "create_dataloaders",
    "train_model",
    "evaluate_model",
    "backtest",
    "PriceSequenceGRU",
    "TimeSeriesMarketDataset",
    "create_ts_dataloaders",
    "evaluate_ts_model",
    "train_ts_model",
]


def __getattr__(name: str):
    if name == "MarketValueNet":
        from .architecture import MarketValueNet

        return MarketValueNet
    if name in {"PolymarketDataset", "create_dataloaders"}:
        from .dataset import PolymarketDataset, create_dataloaders

        return {
            "PolymarketDataset": PolymarketDataset,
            "create_dataloaders": create_dataloaders,
        }[name]
    if name in {"train_model"}:
        from .train import train_model

        return train_model
    if name in {"evaluate_model", "backtest"}:
        from .evaluate import evaluate_model, backtest

        return {
            "evaluate_model": evaluate_model,
            "backtest": backtest,
        }[name]
    if name == "PriceSequenceGRU":
        from .ts_architecture import PriceSequenceGRU

        return PriceSequenceGRU
    if name in {"TimeSeriesMarketDataset", "create_ts_dataloaders"}:
        from .ts_dataset import TimeSeriesMarketDataset, create_ts_dataloaders

        return {
            "TimeSeriesMarketDataset": TimeSeriesMarketDataset,
            "create_ts_dataloaders": create_ts_dataloaders,
        }[name]
    if name == "evaluate_ts_model":
        from .ts_evaluate import evaluate_ts_model

        return evaluate_ts_model
    if name == "train_ts_model":
        from .ts_train import train_ts_model

        return train_ts_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
