"""Carga y validación de configuración desde config.yaml."""

from pathlib import Path

import yaml


def load_config(config_path: str = "config/config.yaml") -> dict:
    """
    Carga configuración desde un archivo YAML.

    Args:
        config_path: Ruta al archivo config.yaml.

    Returns:
        Diccionario con la configuración.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config
