from typing import Set

from app.config import STOPWORD_FILE_PATH


def get_stopwords() -> Set:
    stop_word = set()
    file = open(STOPWORD_FILE_PATH, "r")
    while True:
        line = file.readline()
        if not line:
            break
        stop_word.add(line.strip())
    return stop_word
