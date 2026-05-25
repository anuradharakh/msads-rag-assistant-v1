import json
from pathlib import Path
from typing import Any, List


def write_json(data: Any, output_path: str) -> None:
    """WRITE JSON FILE. **"""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def load_json(input_path: str) -> Any:
    """LOAD JSON FILE. **"""

    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {input_path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_jsonl(records: List[Any], output_path: str) -> None:
    """WRITE JSONL FILE. **"""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(input_path: str) -> List[Any]:
    """READ JSONL FILE. **"""

    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {input_path}")

    records = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            records.append(json.loads(line))

    return records