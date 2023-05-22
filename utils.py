import json
from pathlib import Path
from typing import Generator
from nltk import word_tokenize
import pandas as pd


def read_stopwords(path: Path = Path("./data/czech_stopwords.json")) -> list[str]:
        with open(path) as word_file:
            return json.load(word_file)

def read_files(path: Path = Path("./data/dataset/vyber"),stop_words: list[str] = read_stopwords()) -> Generator:
    for file in path.iterdir():
        with open(file) as text:
            words = word_tokenize(text.read(),"czech")
            yield [word for word in words if word.isalpha() and word.lower() not in stop_words], file.stem.split('_')[1:]

def encode_categories(categories: list[list[str]]) -> tuple:
    counter = -1
    encoding = {}
    encoded_categories = []

    for document in categories:
        document_categories = []

        for category in document:
            if category not in encoding:
                encoding[category] = counter = counter + 1
            document_categories.append(encoding[category])

        encoded_categories.append(document_categories)
            
    return encoded_categories, encoding

def make_dataset(folder_path: Path = Path("./data/dataset/vyber"), stop_words: list[str] = read_stopwords()):
    texts = []
    categories = []

    for text, category in read_files(path = folder_path, stop_words = stop_words):
        texts.append(" ".join(text))
        categories.append(category)

    categories, encoding = encode_categories(categories)
    encoding = pd.DataFrame.from_dict(encoding,orient='index', columns=["code"])
    encoding = encoding.reset_index(names="category name")
    encoding = encoding.set_index("code")

    return pd.DataFrame({"text": texts, "category": categories}), encoding


# def count()