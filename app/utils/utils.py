from pathlib import Path
from typing import Set

from app.config import STOPWORD_FILE_PATH


def get_stopwords() -> Set:
    return set(Path(STOPWORD_FILE_PATH).read_text(encoding="UTF-8").split("\n"))
