import gin

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from simmc.data.os_dataset import OSDataset
from simmc.model.osnet import OSNet, calc_object_similarity


@gin.configurable
class PreTrainEngine(pl.LightningModule):
    def __init__(self, train_dataset: str, val_dataset: str,
        batch_size: int = 32, lr: float = 1e-3, weight_decay: float = 1e-5):
        super(PreTrainEngine, self).__init__()

        self.model = OSNet()
        self.loss = nn.MultiLabelSoftMarginLoss(reduce=False)

        self.train_dataset_path = train_dataset
        self.val_dataset_path = val_dataset

        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay

    @property
    def train_dataset(self):
        return OSDataset(self.train_dataset_path)
    
    @property
    def val_dataset(self):
        return OSDataset(self.val_dataset_path)

    def training_step(self, batch):
        print(batch)

    def configure_optimizers(self):
        opt = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sch = CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2, eta_min=1e-8)
        return [opt], [sch]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
