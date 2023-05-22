from pytorch_lightning import LightningDataModule

class Dataloader(LightningDataModule):

    def __init__(self) -> None:
        super().__init__()