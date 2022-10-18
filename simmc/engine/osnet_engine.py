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

from simmc.data.os_dataset import OSDataset
from simmc.model.osnet import create_tokenizer, OSNet, calc_object_similarity


class collate_fn:
    def __init__(self):
        self.tokenizer = create_tokenizer()

    def __call__(self, samples):
        def proc(key):
            values = [sample[key] for sample in samples]

            if key == "context":
                return self.tokenizer(values, padding=True, truncation=True, return_tensors="pt")

            return torch.stack(values)

        return { k: proc(k) for k in samples[0] }


def focal_loss(inputs: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2, reduce: bool = True):
    p = torch.sigmoid(inputs)
    p_t = p * targets + (1 - p) * (1 - targets)

    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none") * ((1. - p_t) ** gamma)

    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * loss

    if reduce:
        return loss.mean()
    return loss


@gin.configurable
class OSNetEngine(pl.LightningModule):
    def __init__(self, train_dataset: str, val_dataset: str,
        batch_size: int = 32, lr: float = 1e-3, bert_lr: float = 1e-6, weight_decay: float = 1e-5):
        super(OSNetEngine, self).__init__()

        self.model = OSNet()

        self.train_dataset_path = train_dataset
        self.val_dataset_path = val_dataset

        self.batch_size = batch_size
        self.lr = lr
        self.bert_lr = bert_lr
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
        disamb_objects = batch["disamb_objects"].float()

        acts = batch["acts"]
        is_req = batch["is_request"]
        slots = batch["slots"].float()

        object_exists = batch["object_exists"].float()
        labels = batch["labels"].float()

        outputs = self.model(context, objects, object_masks)

        loss_disamb = focal_loss(outputs.disamb, disamb)

        loss_disamb_obj = focal_loss(outputs.disamb_objs, disamb_objects, reduce=False)
        loss_disamb_obj = (loss_disamb_obj * disamb.unsqueeze(-1)).mean()

        loss_act = F.cross_entropy(outputs.acts, acts)
        loss_is_req = focal_loss(outputs.is_request, is_req)
        loss_slots = focal_loss(outputs.slots, slots)

        loss_object_exists = focal_loss(outputs.object_exists, object_exists)
        loss_label = focal_loss(outputs.objects, labels, reduce=False)
        loss_label = (loss_label * object_exists.unsqueeze(-1)).mean()

        loss = loss_object_exists + loss_label + loss_disamb + loss_disamb_obj + loss_act + loss_is_req + loss_slots

        self.log("train_loss", loss.item())
        self.log("train_loss_disamb", loss_disamb.item())
        self.log("train_loss_disamb_obj", loss_disamb_obj.item())
        self.log("train_loss_act", loss_act.item())
        self.log("train_loss_is_req", loss_is_req.item())
        self.log("train_loss_slots", loss_slots.item())
        self.log("train_loss_object_exists", loss_object_exists.item())
        self.log("train_loss_label", loss_label.item())

        return loss

    def validation_step(self, batch, batch_idx):
        context = batch["context"]
        objects = batch["objects"]
        object_masks = batch["object_masks"]

        disamb = batch["disamb"]
        disamb_objects = batch["disamb_objects"].float()

        acts = batch["acts"]
        is_req = batch["is_request"]
        slots = batch["slots"].float()

        object_exists = batch["object_exists"].float()
        labels = batch["labels"].float()

        outputs = self.model(context, objects, object_masks)

        loss_disamb = F.binary_cross_entropy_with_logits(outputs.disamb, disamb)

        loss_disamb_obj = F.binary_cross_entropy_with_logits(outputs.disamb_objs, disamb_objects, reduce=False)
        loss_disamb_obj = (loss_disamb_obj * disamb.unsqueeze(-1)).mean()

        loss_act = F.cross_entropy(outputs.acts, acts)
        loss_is_req = F.binary_cross_entropy_with_logits(outputs.is_request, is_req)
        loss_slots = F.binary_cross_entropy_with_logits(outputs.slots, slots)

        loss_object_exists = F.binary_cross_entropy_with_logits(outputs.object_exists, object_exists)
        loss_label = F.binary_cross_entropy_with_logits(outputs.objects, labels, reduce=False)
        loss_label = (loss_label * object_exists.unsqueeze(-1)).mean()

        loss = loss_object_exists + loss_label + loss_disamb + loss_disamb_obj + loss_act + loss_is_req + loss_slots

        self.log("val_loss", loss.item(), prog_bar=True)
        self.log("val_loss_disamb", loss_disamb.item())
        self.log("val_loss_disamb_obj", loss_disamb_obj.item())
        self.log("val_loss_act", loss_act.item())
        self.log("val_loss_is_req", loss_is_req.item())
        self.log("val_loss_slots", loss_slots.item())
        self.log("val_loss_object_exists", loss_object_exists.item())
        self.log("val_loss_label", loss_label.item())

    def configure_optimizers(self):
        low_lr_parameters = [
            self.model.bert
        ]

        high_lr_parameters = [
            self.model.context_proj,
            self.model.object_feat,
            self.model.os_trans,
            self.model.object_classifier,
            self.model.object_head,
            self.model.disamb_classifier,
            self.model.disamb_head,
            self.model.act_classifier,
            self.model.is_req_classifier,
            self.model.slot_classifier
        ]

        opt_elements = []
        opt_elements.extend([
            { "params": m.parameters(), "lr": self.bert_lr, "weight_decay": self.weight_decay }
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
