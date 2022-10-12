import gin

import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_only

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import mlflow
import mlflow.pytorch

from transformers import AlbertTokenizer

from simmc.data.os_dataset import OSDataset
from simmc.model.osnet import OSNet, calc_object_similarity


class collate_fn:
    def __init__(self):
        self.tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")

    def __call__(self, samples):
        def proc(key):
            values = [sample[key] for sample in samples]

            if key == "context":
                return self.tokenizer(values, padding=True, truncation=True, return_tensors="pt")

            return torch.stack(values)

        return { k: proc(k) for k in samples[0] }


@gin.configurable
class OSNetEngine(pl.LightningModule):
    def __init__(self, train_dataset: str, val_dataset: str,
        batch_size: int = 32, lr: float = 1e-3, weight_decay: float = 1e-5):
        super(OSNetEngine, self).__init__()

        self.model = OSNet()

        self.train_dataset_path = train_dataset
        self.val_dataset_path = val_dataset

        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay

        self.collate_fn = collate_fn()

        self.max_val_loss = None

    @property
    def train_dataset(self):
        return OSDataset(self.train_dataset_path)
    
    @property
    def val_dataset(self):
        return OSDataset(self.val_dataset_path)

    def training_step(self, batch, batch_idx):
        context = batch["context"]
        objects = batch["objects"]
        object_masks = batch["object_masks"]

        disamb = batch["disamb"]
        disamb_objects = batch["disamb_objects"]

        acts = batch["acts"]
        is_req = batch["is_request"]
        slots = batch["slots"]

        labels = batch["labels"]

        outputs = self.model(context, objects)

        pred_label = calc_object_similarity(outputs.context_proj, outputs.object_proj)
        pred_label.masked_fill_(object_masks, -1e4)

        pred_disambs = calc_object_similarity(outputs.context_proj, outputs.disamb_proj)
        pred_disambs.masked_fill_(object_masks, -1e4)

        loss_disamb = F.binary_cross_entropy_with_logits(outputs.disamb, disamb)
        loss_disamb_obj = F.multilabel_soft_margin_loss(pred_disambs, disamb_objects, reduce=False)
        loss_disamb_obj = loss_disamb_obj * disamb.unsqueeze(-1)
        loss_disamb_obj = loss_disamb_obj.mean()

        loss_act = F.cross_entropy(outputs.acts, acts)
        loss_is_req = F.binary_cross_entropy_with_logits(outputs.is_request, is_req)
        loss_slots = F.multilabel_soft_margin_loss(outputs.slots, slots)

        loss_label = F.multilabel_soft_margin_loss(pred_label, labels)

        loss = loss_label + loss_disamb + loss_disamb_obj + loss_act + loss_is_req + loss_slots

        mlflow.log_metric("train_loss", loss.item(), step=self.global_step)

        return loss

    def on_validation_start(self):
        self.val_loss = 0

    def validation_step(self, batch, batch_idx):
        context = batch["context"]
        objects = batch["objects"]
        object_masks = batch["object_masks"]

        disamb = batch["disamb"]
        disamb_objects = batch["disamb_objects"]

        acts = batch["acts"]
        is_req = batch["is_request"]
        slots = batch["slots"]

        labels = batch["labels"]

        outputs = self.model(context, objects)

        pred_label = calc_object_similarity(outputs.context_proj, outputs.object_proj)
        pred_label.masked_fill_(object_masks, -1e4)

        pred_disambs = calc_object_similarity(outputs.context_proj, outputs.disamb_proj)
        pred_disambs.masked_fill_(object_masks, -1e4)

        loss_disamb = F.binary_cross_entropy_with_logits(outputs.disamb, disamb)
        loss_disamb_obj = F.multilabel_soft_margin_loss(pred_disambs, disamb_objects, reduce=False)
        loss_disamb_obj = loss_disamb_obj * disamb.unsqueeze(-1)
        loss_disamb_obj = loss_disamb_obj.mean()

        loss_act = F.cross_entropy(outputs.acts, acts)
        loss_is_req = F.binary_cross_entropy_with_logits(outputs.is_request, is_req)
        loss_slots = F.multilabel_soft_margin_loss(outputs.slots, slots)

        loss_label = F.multilabel_soft_margin_loss(pred_label, labels)

        loss = loss_label + loss_disamb + loss_disamb_obj + loss_act + loss_is_req + loss_slots

        loss = loss.item()
        self.val_loss = (self.val_loss * batch_idx + loss) / (batch_idx + 1)

    @rank_zero_only
    def on_validation_end(self):
        mlflow.log_metric("val_loss", self.val_loss, step=self.global_step)

        log_model = True 

        if self.max_val_loss:
            if self.max_val_loss < self.val_loss:
                log_model = False

        if log_model:
            self.max_val_loss = self.val_loss
            print("new best model (val_loss: {:.4f})".format(self.val_loss))
            mlflow.pytorch.log_model(self.model, "model")
        else:
            print("model improve failed (cur: {:.4f}, best: {:.4f})".format(self.val_loss, self.max_val_loss))
            

    def configure_optimizers(self):
        low_lr_parameters = [
            self.model.bert
        ]

        high_lr_parameters = [
            self.model.context_proj,
            self.model.object_feat,
            self.model.object_proj,
            self.model.disamb_classifier,
            self.model.disamb_proj,
            self.model.act_classifier,
            self.model.is_req_classifier,
            self.model.slot_classifier
        ]

        opt_elements = []
        opt_elements.extend([
            { "params": m.parameters(), "lr": self.lr * 1e-3, "weight_decay": self.weight_decay }
            for m in low_lr_parameters])
        opt_elements.extend([
            { "params": m.parameters(), "lr": self.lr, "weight_decay": self.weight_decay }
            for m in high_lr_parameters])

        opt = optim.AdamW(opt_elements)
        sch = CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2, eta_min=1e-8)
        return [opt], [sch]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
            collate_fn=self.collate_fn, num_workers=4, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
            collate_fn=self.collate_fn, num_workers=4, pin_memory=True)
