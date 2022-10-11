import gin

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from transformers import AlbertTokenizer

from simmc.data.os_dataset import OSDataset
from simmc.model.osnet import OSNet, calc_object_similarity


class collate_fn:
    def __init__(self):
        self.tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")

    def __call__(self, samples):
        context = [sample["context"] for sample in samples]
        objects = [sample["objects"] for sample in samples]
        labels = [sample["labels"] for sample in samples]
        object_masks = [sample["object_masks"] for sample in samples]

        context = self.tokenizer(context, padding=True, truncation=True, return_tensors="pt")
        objects = torch.stack(objects)
        labels = torch.stack(labels)
        object_masks = torch.stack(object_masks)

        return {
            "context": context,
            "objects": objects,
            "labels": labels,
            "object_masks": object_masks
        }


@gin.configurable
class PreTrainEngine(pl.LightningModule):
    def __init__(self, train_dataset: str, val_dataset: str,
        batch_size: int = 32, lr: float = 1e-3, weight_decay: float = 1e-5):
        super(PreTrainEngine, self).__init__()

        self.model = OSNet()
        self.loss = nn.MultiLabelSoftMarginLoss()

        self.train_dataset_path = train_dataset
        self.val_dataset_path = val_dataset

        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay

        self.collate_fn = collate_fn()

    @property
    def train_dataset(self):
        return OSDataset(self.train_dataset_path)
    
    @property
    def val_dataset(self):
        return OSDataset(self.val_dataset_path)

    def training_step(self, batch, batch_idx):
        context = batch["context"]
        objects = batch["objects"]
        labels = batch["labels"]
        object_masks = batch["object_masks"]

        context_proj, object_proj = self.model(context, objects)
        output = calc_object_similarity(context_proj, object_proj)
        output.masked_fill_(object_masks, -1e5)

        loss = self.loss(output, labels)

        return loss

    def validation_step(self, batch, batch_idx):
        context = batch["context"]
        objects = batch["objects"]
        labels = batch["labels"]
        object_masks = batch["object_masks"]

        context_proj, object_proj = self.model(context, objects)
        output = calc_object_similarity(context_proj, object_proj)
        output.masked_fill(object_masks, -1e5)

        loss = self.loss(output, labels).item()

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        opt = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sch = CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2, eta_min=1e-8)
        return [opt], [sch]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
            collate_fn=self.collate_fn, num_workers=4, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
            collate_fn=self.collate_fn, num_workers=4, pin_memory=True)
