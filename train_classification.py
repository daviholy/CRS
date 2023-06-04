from pathlib import Path

import torch

from _dataclasses import EncodedTextDataset
from data_modules import DocumentDataModule
from model_system import DocumentClassifier

torch.multiprocessing.set_sharing_strategy("file_system")

training_dataset = EncodedTextDataset.load(Path("./data/train_dataset.json"))
test_dataset = EncodedTextDataset.load(Path("./data/test_dataset.json"))

datamodule = DocumentDataModule(
    training_dataset, test_dataset, test_dataset, batch_size=64, num_workers=10
)

classifier = DocumentClassifier.load_from_checkpoint("model_classifier.ckpt")
