import numpy as np
from numpy.typing import NDArray
import torch
from lightning.pytorch import LightningDataModule
from numba import njit
from torch.utils.data import DataLoader

from _dataclasses import EncodedTextDataset


@njit(cache=True, nogil=True, inline="always")
def _create_prob_vector(context: NDArray, unique_word_count: int):
    result = np.zeros(unique_word_count, dtype=np.float32)

    for context_word in context:
        result[context_word] = 1
    return result


@njit(cache=True, nogil=True, inline="always")
def preprocess_document(document: NDArray, unique_word_count: int, context_size: int):
    result_contexts = np.empty((len(document), unique_word_count), dtype=np.float32)

    for document_index, _ in enumerate(document):
        minimal_index = max([0, document_index - context_size])
        maximal_index = min([len(document), document_index + context_size + 1])

        word_context = document[minimal_index:maximal_index]
        result_contexts[document_index] = _create_prob_vector(
            word_context, unique_word_count
        )

    return result_contexts


def _words_collate_fn(
    documents: list[list], unique_word_count: int, context_size: int
) -> tuple:
    word_count = 0
    for _, document in documents:
        word_count += len(document)

    result_contexts = np.empty((word_count, unique_word_count), dtype=np.float32)
    result_words = np.empty(word_count, dtype=np.int32)
    index = 0

    for _, document in documents:
        result_words[index : index + len(document)] = document
        result_contexts[index : index + len(document)] = preprocess_document(
            document, unique_word_count, context_size
        )

        index += len(document)

    return torch.as_tensor(result_words), torch.as_tensor(result_contexts)


class EmbeddingDataModule(LightningDataModule):
    def __init__(
        self,
        Dataset: EncodedTextDataset,
        batch_size: int = 1,
        context_size: int = 2,
        num_workers=16,
    ) -> None:
        super().__init__()
        self._num_workers = num_workers
        self._batch_size = batch_size
        self._training_dataset = Dataset
        self._context_size = context_size

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._training_dataset,
            batch_size=self._batch_size,
            collate_fn=lambda x: _words_collate_fn(
                x, self._training_dataset.unique_words_count(), self._context_size
            ),
            num_workers=self._num_workers,
            shuffle=True,
        )


class DocumentDataModule(LightningDataModule):
    def __init__(
        self,
        training_dataset: EncodedTextDataset,
        validation_dataset: EncodedTextDataset,
        test_dataset: EncodedTextDataset,
        batch_size: int = 1,
        num_workers=16,
    ) -> None:
        super().__init__()
        self._num_workers = num_workers
        self._batch_size = batch_size
        self._training_dataset = training_dataset
        self._validation_dataset = validation_dataset
        self._test_dataset = test_dataset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._training_dataset,
            batch_size=self._batch_size,
            collate_fn=self._collate_fn,
            num_workers=self._num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._validation_dataset,
            batch_size=self._batch_size,
            collate_fn=self._collate_fn,
            num_workers=self._num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test_dataset,
            batch_size=self._batch_size,
            collate_fn=self._collate_fn,
            num_workers=self._num_workers,
        )

    def _collate_fn(self, documents: list) -> tuple:
        tensor_categories = torch.empty(
            len(documents), self._training_dataset.categories_count()
        )
        tensor_documents = []

        for index, input in enumerate(documents):
            categories, document = input
            tensor_categories[index] = torch.as_tensor(
                _create_prob_vector(
                    categories, self._training_dataset.categories_count()
                )
            )
            tensor_documents.append(torch.as_tensor(document))

        return tensor_documents, tensor_categories
