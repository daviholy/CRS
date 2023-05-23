from pytorch_lightning import LightningDataModule
from _dataclasses import TextDataset
from torch.utils.data import DataLoader
from torch import Tensor
import torch

class EmbeddingDataloader(LightningDataModule):

    def __init__(self, Dataset: TextDataset, batch_size: int = 32) -> None:
        super().__init__()
        self._batch_size = batch_size
        self._training_dataset = Dataset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self._training_dataset, batch_size=self._batch_size, collate_fn= self._collate_fn)
    
    def _collate_fn(self, documents: list) -> list[Tensor]:
        result = []

        for document in documents:
            result.append(torch.as_tensor(document.split(" ")))

        return result

    
