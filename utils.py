import json
from pathlib import Path
from typing import Generator

import pandas as pd
import torch
from nltk import word_tokenize

from _dataclasses import TextDataset


def read_stopwords(path: Path = Path("./data/czech_stopwords.json")) -> list[str]:
    with open(path) as word_file:
        return json.load(word_file)


def read_files(
    path: Path = Path("./data/dataset/vyber"),
    stop_words: list[str] = read_stopwords(),
    suffixes: list[str] = [".lemma"],
) -> Generator:
    """
    Read  and filter the files of dataset

    Args:
        path (Path, optional): Path to the dataset folder. Defaults to Path("./data/dataset/vyber").
        stop_words (list[str], optional): stop words vocabulary. Defaults to read_stopwords().
        suffixes (list[str], optional): _description_. Defaults to [".lemma"].

    Yields:
        Generator: _description_
    """
    count = 0
    for file in path.iterdir():
        if file.suffix in suffixes:
            count += 1
            with open(file) as text:
                words = word_tokenize(text.read(), "czech")
                yield [
                    word
                    for word in words
                    if word.isalpha() and word.lower() not in stop_words
                ], file.stem.split("_")[1:]




def make_dataset(
    folder_path: Path = Path("./data/dataset/vyber"),
    stop_words: list[str] = read_stopwords(),
    suffixes: list[str] = [".lemma"],
) -> TextDataset:
    """
    Create dataset from given folder 

    Args:
        folder_path (Path, optional): Path to the dataset folder. Defaults to Path("./data/dataset/vyber").
        stop_words (list[str], optional): Stop words vocabulary. Defaults to read_stopwords().
        suffixes (list[str], optional): file suffixes to consider. Defaults to [".lemma"].

    Returns:
        TextDataset: _description_
    """
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
    
    
    texts = []
    categories = []

    for text, category in read_files(
        path=folder_path, stop_words=stop_words, suffixes=suffixes
    ):
        texts.append(" ".join(text))
        categories.append(category)

    categories, encoding = encode_categories(categories)
    encoding = pd.DataFrame.from_dict(encoding, orient="index", columns=["code"])
    encoding = encoding.reset_index(names="category name")
    encoding["category name"] = encoding["category name"].astype(str)
    encoding["code"] = encoding["code"].astype(int)
    encoding = encoding.set_index("code")

    df = pd.DataFrame({"text": texts, "category": categories})

    return TextDataset(texts=df, category_encoding=encoding)

def sort_confusion_matrices(
    matrices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sort the given confusion matrices.

    Args:
        matrices (torch.Tensor): (n,2,2) Tensor of matrices.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: tuple of sorted values. First is values, second indices
    """
    sum = matrices[:, 1, :].sum(1).sort(descending=True)
    return sum.values, sum.indices


# def count()
