from typing import Any, Callable, Iterable

import pandas as pd
import torch
from aim.pytorch_lightning import AimLogger
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from torch import Tensor
from torch.nn.functional import cross_entropy
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.classification import MultilabelConfusionMatrix, MultilabelFBetaScore

from _dataclasses import EncodedTextDataset
from data_modules import DocumentDataModule, EmbeddingDataModule
from model import Classifier, SkipGram


class word2vec(LightningModule):
    def __init__(self, output_size: int, lr: float = 0.01, embedding_output=100):
        super().__init__()
        self.save_hyperparameters()

        self.model = SkipGram(output_size, embedding_size=embedding_output)

        self.eval()

    def on_train_epoch_start(self):
        self.model.train()

    def training_step(self, x: Tensor):
        result = self.model(x[0])
        loss = cross_entropy(result, x[1], label_smoothing=0.25)
        self.log("train_embedding_loss", loss, on_epoch=False, on_step=True)
        return loss

    def configure_optimizers(self):
        dense_optimizer = AdamW(self.parameters(), lr=self.hparams["lr"])
        return {
            "optimizer": dense_optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(dense_optimizer, patience=1),
                "monitor": "train_embedding_loss",
            },
        }

    def fit(
        self,
        datamodule: EmbeddingDataModule,
        epochs: int = 10,
        log_frequency: int = 10,
        backup_epochs: int = 2,
        backups_top_k: int = 2,
    ) -> None:
        """
        Fit (train) the model.

        Args:
            datamodule (EmbeddingDataModule): Datamodule of training data
            epochs (int, optional): Epochs training. Defaults to 10.
            log_frequency (int, optional): logging step frequency. Defaults to 10.
            backup_epochs (int, optional): Epoch interval between backup checking. Defaults to 2.
            backups_top_k (int, optional): How many backups currently store for this run. Defaults to 2.
        """
        trainer = Trainer(
            logger=AimLogger(),
            max_epochs=epochs,
            limit_val_batches=0,
            num_sanity_val_steps=0,
            log_every_n_steps=log_frequency,
            callbacks=[
                LearningRateMonitor(logging_interval="epoch"),
                ModelCheckpoint(
                    dirpath="backups/",
                    save_top_k=backup_epochs,
                    every_n_epochs=backups_top_k,
                    monitor="train_embedding_loss",
                ),
            ],
        )
        trainer.fit(self, datamodule=datamodule)
        trainer.save_checkpoint("model_embedding.ckpt")

    def embedding_encoded_document(self, document: Tensor) -> Tensor:
        """
        Compute embeddings for the given document.

        Args:
            documents (Tensor): Document for embedding computing.

        Returns:
            Tensor: Document embeddings.
        """
        result = torch.empty(len(document), self.hparams["embedding_output"])
        for index, document in enumerate(document):
            result[index] = self.forward(document).mean(0)
        return result

    def embedding_words(self, dataset: EncodedTextDataset) -> pd.DataFrame:
        """
        Calculate embeddings fow each unique word in dataset.

        Args:
            dataset (EncodedTextDataset): Dataset for embedding calculation

        Returns:
            DataFrame: embeddings of the words
        """
        size = dataset.unique_words_count()

        words_embedding = self.forward(torch.as_tensor(list(range(size))))

        df = pd.DataFrame(words_embedding.detach().numpy())
        df["word"] = dataset.encoding["word"]
        return df[df["word"] != "_"]

    def embedding_encoded_documents(self, dataset: EncodedTextDataset) -> pd.DataFrame:
        """
        Calculate embeddings for dataset  documents.

        Args:
            dataset (EncodedTextDataset): Documents

        Returns:
            pd.DataFrame: Documents embeddings.
        """
        documents_embeddings = []

        for text in dataset.texts["text"]:
            documents_embeddings.append(
                self.embedding_encoded_document(torch.as_tensor(text))[0].detach().numpy()
            )

        df = pd.DataFrame(documents_embeddings)
        df["category"] = dataset.texts["category"]
        return df

    def forward(self, x) -> Any:
        return self.model(x)


class DocumentClassifier(LightningModule):
    def __init__(self, embedder: word2vec, output_size, lr=0.001) -> None:
        super().__init__()
        self.save_hyperparameters()

        self._embedder = embedder
        self._classifier = Classifier(
            self._embedder.hparams["embedding_output"], output_size
        )

        self._embedder.freeze()
        self._embedder.eval()

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams["lr"])
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, patience=5, factor=0.5),
                "monitor": "train_epoch_classification_loss",
            },
        }

    def on_train_epoch_start(self):
        self.train()
        self._embedder.eval()

        self._train_loss_mean = 0
        self._train_loss_mean_count = 0

    def training_step(self, x: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        result = self._embedder.embedding_encoded_document(x[0])
        result = self._classifier(result)
        loss = cross_entropy(result, x[1], label_smoothing=0.3)
        self.log("train_step_classification_loss", loss, batch_size=result.shape[0])
        self.log(
            "train_epoch_classification_loss",
            loss,
            on_epoch=True,
            on_step=False,
            batch_size=result.shape[0],
        )
        self._train_loss_mean += loss
        self._train_loss_mean_count += result.shape[0]
        return loss

    def on_validation_epoch_start(self) -> None:
        self._validation_loss_mean = 0
        self._validation_loss_mean_count = 0

        self.train()
        self._embedder.eval()

    def validation_step(self, x: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        result = self._embedder.embedding_encoded_document(x[0])
        result = self._classifier(result)
        loss = cross_entropy(result, x[1], label_smoothing=0.25)
        if not self.trainer.sanity_checking:
            self.log("val_epoch_classification_loss", loss, batch_size=result.shape[0])
            self._validation_loss_mean += loss
            self._validation_loss_mean_count += result.shape[0]
        return loss

    def on_validation_epoch_end(self) -> None:
        if not self.trainer.sanity_checking:
            self.log(
                "val_epoch_val-train_classification_loss",
                self._validation_loss_mean / self._validation_loss_mean_count
                - self._train_loss_mean / self._train_loss_mean_count,
            )

    def on_test_epoch_start(self) -> None:
        # TN FP
        # FN TP
        self._confusion_matrix = MultilabelConfusionMatrix(self._output_labels_size, threshold=self._split.item())  # type: ignore
        self._confusion_matrix_normalized = MultilabelConfusionMatrix(
            self._output_labels_size,
            threshold=self._split.item(),  # type: ignore
            normalize="true",
        )

    def test_step(self, x: tuple, batch_idx: int):
        result = self._embedder.embedding_encoded_document(x[0])
        result = self._classifier(result)
        self._confusion_matrix(result, x[1].to(torch.int))
        self._metric(result, x[1].to(torch.int))
        self._confusion_matrix_normalized(result, x[1].to(torch.int))

        return result

    def fit(
        self,
        datamodule: DocumentDataModule,
        epochs: int = 10,
        log_frequency: int = 10,
        backup_epochs: int = 2,
        backups_top_k: int = 2,
    ) -> None:
        trainer = Trainer(
            logger=AimLogger(),
            max_epochs=epochs,
            log_every_n_steps=log_frequency,
            callbacks=[
                LearningRateMonitor(logging_interval="epoch", log_momentum=True),
                ModelCheckpoint(
                    dirpath="backups/",
                    save_top_k=backup_epochs,
                    every_n_epochs=backups_top_k,
                    monitor="val_epoch_classification_loss",
                ),
            ],
        )
        trainer.fit(self, datamodule=datamodule)
        trainer.save_checkpoint("model_classifier.ckpt")

    def test(
        self,
        datamodule: DocumentDataModule,
        metric: Callable,
        split: float = 0.5,
        trainer: Trainer | None = None,
    ):
        if trainer is None:
            trainer = Trainer(logger=AimLogger())
        self._split = split
        self._metric = metric
        self._output_labels_size = datamodule._test_dataset.categories_count()

        trainer.test(self, datamodule=datamodule)
        return (
            self._metric.compute(),
            self._confusion_matrix.compute(),
            self._confusion_matrix_normalized.compute(),
        )

    def find_best_decision_split(
        self,
        datamodule: DocumentDataModule,
        beta: float,
        ranges: Iterable = torch.linspace(0, 1, 20),
    ):
        score = 0
        matricies = torch.empty(0)
        matrices_normalized = torch.empty(0)
        best_split = 0
        trainer = Trainer(logger=AimLogger(), enable_progress_bar=False)

        for split in ranges:
            print(f"split: {split.item()}")
            (
                current_score,
                current_matricies,
                current_matricies_normalized,
            ) = self.test(
                datamodule=datamodule,
                split=split,
                metric=MultilabelFBetaScore(
                    beta,
                    datamodule._test_dataset.categories_count(),
                    threshold=split.item(),
                ),
                trainer=trainer,
            )
            print(f"score:{current_score.item()}")
            print()
            if current_score > score:
                score = current_score
                matricies = current_matricies
                best_split = split.item()
                matrices_normalized = current_matricies_normalized

        return matricies, matrices_normalized, best_split, score
