from typing import Self
import numpy as np
from dataclasses import dataclass
from math import floor
from pathlib import Path
from pandas import DataFrame, read_csv

@dataclass
class Dataset():
    texts:  DataFrame
    encoding: DataFrame

    def save(self, texts_path: Path = Path("./data/dataset.csv"), encoding_path = Path("./data/encoding.csv")):
        self.texts.to_csv(texts_path, index= False)
        self.encoding.to_csv(encoding_path, index = False)
    
    @staticmethod
    def load(texts_path: Path = Path("./data/dataset.csv"), encoding_path = Path("./data/categories_encoding.csv")):
        return Dataset(texts = read_csv(texts_path), encoding= read_csv(encoding_path))
    
    def count_words(self) -> DataFrame:
        words = {}
        for text in self.texts["text"]:
            for word in text.split(" "):
                try:
                    words[word] += 1
                except Exception:
                    words[word] = 1
        return DataFrame.from_dict(words, orient = "index", columns=["count"])
    
    def filter_minimal_word_count(self, minimum: int = 20) -> None:
        counted_words = self.count_words()["count"]

        for index in range(self.texts.shape[0]):
            sentence = self.texts.iloc[index]["text"].split(" ")

            for index_word, word in enumerate(sentence):
                if counted_words[word] <= minimum:
                    sentence[index_word] = "_"
            
            self.texts.iloc[index]["text"] = " ".join(sentence)
    
    def split(self, frac: float = 0.2) -> Self:
        splitted_dataset = Dataset(texts= self.texts.sample(frac = frac ), encoding = self.encoding)
        self.texts.drop(splitted_dataset.texts.index, inplace=True) #type: ignore
        return splitted_dataset
