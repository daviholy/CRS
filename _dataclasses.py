"""dataclasses

Dataclasses which stores data and define basic operations on it.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
from pandas import DataFrame, read_json
from torch import Tensor
from torch.utils.data import Dataset


@dataclass
class train_token:
    input_word: Tensor
    result: Tensor


@dataclass
class TextDataset(Dataset):
    texts: DataFrame
    category_encoding: DataFrame

    def save(
        self,
        texts_path: Path = Path("./data/dataset.json"),
        category_encoding_path: Path = Path("./data/category_encoding.json"),
    ):
        """Save currently loaded dataset on disk in csv format.

        Args:
            encoding_path (Path): Saving path for encoding mapping
            texts_path (Path): Saving path for texts.
        """
        self.texts.to_json(texts_path, orient="values")
        self.category_encoding.to_json(category_encoding_path, orient="values")

    @staticmethod
    def load(
        texts_path: Path = Path("./data/dataset.json"),
        category_encoding_path: Path=Path("./data/category_encoding.json"),
    ) -> "TextDataset":
        """
        Load the state from Files previously saved.

        Args:
            texts_path (Path, optional): Path to the storing file of text . Defaults to Path("./data/dataset.json").
            category_encoding_path (Path, optional): Path to the file which will store categories. Defaults to Path("./data/category_encoding.json").
        """
        loaded = TextDataset(
            texts=read_json(texts_path),
            category_encoding=read_json(category_encoding_path),
        )

        loaded.texts = loaded.texts.rename(columns={0: "text", 1: "category"})
        loaded.category_encoding = loaded.category_encoding.rename(columns={0: "word"})

        loaded.texts["text"] = loaded.texts["text"].apply(lambda x: np.array(x, dtype=np.int32))  # type: ignore
        loaded.texts["category"] = loaded.texts["category"].apply(lambda x: np.array(x, dtype=np.int32))  # type: ignore
        return loaded

    def count_words(self) -> DataFrame:
        """
        Count the occurences of words.

        Returns:
            DataFrame: occurence of each word in dataset
        """
        words = {}
        for text in self.texts["text"]:
            for word in text.split(" "):
                try:
                    words[word] += 1
                except Exception:
                    words[word] = 1
        return DataFrame.from_dict(words, orient="index", columns=["count"])

    def filter_minimal_word_count(self, minimum: int = 20) -> Self:
        """
        Erase (replace with _) words under the specified minimum.

        Args:
            minimum (int, optional): Minimal occurences of owrd. Defaults to 20.

        Returns:
            Self: filtered dataset
        """
        counted_words = self.count_words()["count"]
        documents = []

        for index in range(len(self)):
            sentence = self.texts.iloc[index]["text"].split(" ")

            for index_word, word in enumerate(sentence):
                if counted_words[word] <= minimum:
                    sentence[index_word] = "_"

            documents.append(" ".join(sentence))

        return TextDataset(
            texts=DataFrame({"text": documents, "category": self.texts["category"]}),
            category_encoding=self.category_encoding,
        )

    def decode_categories(self, dataset: DataFrame | None = None) -> DataFrame:
        """
        Decode the numerically encoded categories.

        Args:
            dataset (DataFrame | None, optional): . Defaults to None.

        Returns:
            DataFrame: _description_
        """
        if dataset is None:
            dataset = self.texts
        dataset["category"] = dataset["category"].map(self.category_encoding["word"])
        return dataset

    def split(self, frac: float = 0.2) -> Self:
        """
        Split the dataset. return the new one and drop it in current

        Args:
            frac (float, optional): Fract of which drop. Defaults to 0.2.

        Returns:
            Self: New splitted dataframe.
        """
        splitted_dataset = TextDataset(
            texts=self.texts.sample(frac=frac), category_encoding=self.category_encoding
        )
        self.texts.drop(splitted_dataset.texts.index, inplace=True)  # type: ignore
        return splitted_dataset

    def encode(self) -> "EncodedTextDataset":
        """
        Encode the texts.

        Returns:
            EncodedTextDataset: encoded texts
        """
        converts = {}
        counter = -1
        documents = []

        for index in range(len(self)):
            document = []

            for word in self.texts.iloc[index]["text"].split(" "):
                code = converts.get(word)
                if code == None:
                    code = (counter := counter + 1)
                    converts[word] = code
                document.append(code)

            documents.append(document)
        dataset = EncodedTextDataset(
            texts=DataFrame({"texts": documents, "category": self.texts["category"]}),
            category_encoding=self.category_encoding,
            encoding=DataFrame.from_dict(
                data=converts, orient="index", columns=["code"]
            ),
        )
        dataset.encoding.index.name = "word"
        dataset.encoding.set_index("code")
        return dataset

    def explode(self) -> Self:
        """Explode category.

        Transform categories into single values by cloning the rows.

        Returns:
            Self: new Dataframe
        """
        return TextDataset(
            texts=self.texts.explode("category"),
            category_encoding=self.category_encoding,
        )

    def __len__(self):
        return self.texts.shape[0]

    def __getitem__(self, index: int) -> str:
        return self.texts.iloc[index]["text"]


@dataclass
class EncodedTextDataset(TextDataset):
    encoding: DataFrame

    @staticmethod
    def load(
        texts_path: Path = Path("./data/dataset.json"),
        category_encoding_path=Path("./data/category_encoding.json"),
        encoding_path=Path("./data/encoding.json"),
    ) -> "EncodedTextDataset":
        """
        Load the state from Files previously saved.

        Args:
            texts_path (Path, optional): Path to the storing json file. Defaults to Path("./data/dataset.json").
            category_encoding_path (Path, optional): Path to the storing category json file. Defaults to Path("./data/category_encoding.json").
            encoding_path (Path, optional): Path to storing encoding file. Defaults to Path("./data/encoding.json").

        Returns:
            EncodedTextDataset: loaded dataframe
        """
        loaded = TextDataset.load(
            texts_path=texts_path, category_encoding_path=category_encoding_path
        )
        loaded = EncodedTextDataset(
            texts=loaded.texts,
            category_encoding=loaded.category_encoding,
            encoding=read_json(encoding_path),
        )
        # loaded.encoding.rename(columns={0:"word"})
        loaded.encoding.index.name = "word"
        loaded.encoding["code"] = loaded.encoding["code"].apply(lambda x: np.array(x, dtype=np.int32))  # type: ignore
        loaded.encoding = loaded.encoding.reset_index()
        loaded.encoding = loaded.encoding.set_index("code")
        return loaded

    def save(
        self,
        texts_path: Path = Path("./data/dataset.json"),
        category_encoding_path: Path = Path("./data/category_encoding.json"),
        encoding_path: Path = Path("./data/encoding.json"),
    ) -> None:
        """
        Save the current state of dataframe to disk

        Args:
            texts_path (Path, optional): Path to the storing json file. Defaults to Path("./data/dataset.json").
            category_encoding_path (Path, optional): Path to the storing category json file. Defaults to Path("./data/category_encoding.json").
            encoding_path (Path, optional): Path to storing encoding file. Defaults to Path("./data/encoding.json").
        """
        super().save(
            texts_path=texts_path, category_encoding_path=category_encoding_path
        )
        self.encoding.to_json(encoding_path)

    def unique_words_count(self) -> int:
        """unique words

        Returns:
            int: count of unique words 
        """
        return self.encoding.shape[0]

    def categories_count(self) -> int:
        """Number of categories.

        Returns:
            int: category count
        """
        return self.category_encoding.shape[0]

    def explode(self) -> Self:
        """Explode category.

        Transform categories into single values by cloning the rows.

        Returns:
            Self: new Dataframe
        """
        return EncodedTextDataset(
            texts=self.texts.explode("category").reset_index(),
            category_encoding=self.category_encoding,
            encoding=self.encoding,
        )

    def __len__(self):
        return self.texts.shape[0]

    def __getitem__(self, index: int) -> tuple:
        return self.texts.iloc[index]["category"], self.texts.iloc[index]["text"]
