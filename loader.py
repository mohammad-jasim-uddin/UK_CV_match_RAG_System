from pathlib import Path


def load_text_file(file_path: str) -> str:
    """Load a UTF-8 text file and return its content."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    return path.read_text(encoding="utf-8")
