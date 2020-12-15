from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader


class DOOMDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def prepare_data(self, *args, **kwargs):
        # TODO: consider calling the sequence cutter from here.
        # TODO: possibly, download the cocodoom dataset here.
        pass

    def setup(self, stage: Optional[str] = None):
        pass

    def train_dataloader(self, ) -> DataLoader:
        pass
