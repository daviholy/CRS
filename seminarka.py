from pathlib import Path
from typing import Iterable
from nltk import word_tokenize
import json

STOPWORDS_DICT = "./czech_stopwords.json"
stop_words = []

with open(STOPWORDS_DICT) as word_file:
    stop_words = json.load(word_file)
    
def read_files(path: Path = Path("./data/upraveno")) -> Iterable[list[str]]:
    for file in path.iterdir():
        with open(file) as text:
            words = word_tokenize(text,"czech")
            yield [word for word in words if word.lower() not in stop_words]

