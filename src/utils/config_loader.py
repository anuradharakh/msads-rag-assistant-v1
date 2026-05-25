import yaml
from pathlib import Path
from typing import Dict


def load_config(config_path: str = "config.yaml") -> Dict:
    """LOAD YAML CONFIG FILE. **"""

    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config