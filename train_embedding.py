from _dataclasses import EncodedTextDataset
from data_modules import EmbeddingDataModule
from model_system import word2vec

dataset = EncodedTextDataset.load()

embedding = word2vec(dataset.unique_words_count(), lr=0.025)

embedding.fit(
    datamodule=EmbeddingDataModule(
        dataset, batch_size=32, context_size=4, num_workers=13
    ),
    epochs=10,
    log_frequency=100,
)
