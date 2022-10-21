import gin

import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import mlflow
import mlflow.pytorch

from simmc.data.metadata import MetadataDB
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


def focal_loss(inputs: torch.Tensor, targets: torch.Tensor, weight: torch.Tensor = None, alpha: float = 0.25, gamma: float = 2, reduce: bool = True):
    p = torch.sigmoid(inputs)
    p_t = p * targets + (1 - p) * (1 - targets)

    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none", weight=weight) * ((1. - p_t) ** gamma)

    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * loss

    if reduce:
        return loss.mean()
    return loss


@gin.configurable
class OSNetEngine(pl.LightningModule):
    def __init__(self, train_dataset: str, val_dataset: str, metadata_db_path: str, act_ce_weight: str = None,
        batch_size: int = 32, lr: float = 1e-3, bert_lr: float = 1e-6, weight_decay: float = 1e-5):
        super(OSNetEngine, self).__init__()

        self.db = MetadataDB(metadata_db_path)

        self.model = OSNet(self.db)

        self.train_dataset_path = train_dataset
        self.val_dataset_path = val_dataset

        if act_ce_weight:
            self.act_ce_weight = self.register_buffer("act_ce_weight", torch.load(act_ce_weight))

        self.pos_weight = self.register_buffer("pos_weight", torch.tensor(100.))

        self.batch_size = batch_size
        self.lr = lr
        self.bert_lr = bert_lr
        self.weight_decay = weight_decay

        self.collate_fn = collate_fn()

        self.max_val_loss = None

    @property
    def train_dataset(self):
        return OSDataset(self.train_dataset_path, shuffle_objects=True)
    
    @property
    def val_dataset(self):
        return OSDataset(self.val_dataset_path)

    def training_step(self, batch, batch_idx):
        context = batch["context"]
        object_map = batch["object_map"]
        object_masks = batch["object_masks"]

        disamb = batch["disamb"]
        disamb_objects = batch["disamb_objects"].float()

        acts = batch["acts"]

        request_slot_exist = batch["request_slot_exist"]
        request_slot = batch["request_slot"].float()

        object_exist = batch["object_exist"]
        objects = batch["objects"].float()

        slot_query = batch["slot_query"].float()

        outputs = self.model(context, object_map, object_masks)

        object_masks = object_masks.float()

        loss_disamb = focal_loss(outputs.disamb, disamb)

        loss_disamb_obj = F.binary_cross_entropy_with_logits(outputs.disamb_objs, disamb_objects, reduce=False, weight=(1 - object_masks), pos_weight=self.pos_weight)
        loss_disamb_obj = (loss_disamb_obj * disamb.unsqueeze(-1)).mean()

        if hasattr(self, "act_ce_weight"):
            loss_act = F.cross_entropy(outputs.acts, acts, weight=self.act_ce_weight, label_smoothing=0.1)
        else:
            loss_act = F.cross_entropy(outputs.acts, acts, label_smoothing=0.1)

        loss_request_slot_exist = focal_loss(outputs.request_slot_exist, request_slot_exist)
        loss_request_slot = focal_loss(outputs.request_slot, request_slot, reduce=False)
        loss_request_slot = (loss_request_slot * request_slot_exist).mean()

        loss_object_exist = focal_loss(outputs.object_exist, object_exist)
        loss_objects = F.binary_cross_entropy_with_logits(outputs.objects, objects, reduce=False, weight=(1 - object_masks), pos_weight=self.pos_weight)
        loss_objects = (loss_objects * object_exist.unsqueeze(-1)).mean()

        loss_slot_query = focal_loss(outputs.slot_query, slot_query)

        loss = loss_disamb + loss_disamb_obj + \
            loss_act + loss_request_slot_exist + loss_request_slot + \
            loss_object_exist + loss_objects + loss_slot_query

        self.log("train_loss", loss.item())
        self.log("train_loss_disamb", loss_disamb.item())
        self.log("train_loss_disamb_obj", loss_disamb_obj.item())
        self.log("train_loss_act", loss_act.item())
        self.log("train_loss_request_slot_exist", loss_request_slot_exist.item())
        self.log("train_loss_request_slot", loss_request_slot.item())
        self.log("train_loss_object_exist", loss_object_exist.item())
        self.log("train_loss_objects", loss_objects.item())
        self.log("train_loss_slot_query", loss_slot_query.item())

        return loss

    def validation_step(self, batch, batch_idx):
        context = batch["context"]
        object_map = batch["object_map"]
        object_masks = batch["object_masks"]

        disamb = batch["disamb"]
        disamb_objects = batch["disamb_objects"].float()

        acts = batch["acts"]

        request_slot_exist = batch["request_slot_exist"]
        request_slot = batch["request_slot"].float()

        object_exist = batch["object_exist"]
        objects = batch["objects"].float()

        slot_query = batch["slot_query"].float()

        outputs = self.model(context, object_map, object_masks)

        object_masks = object_masks.float()

        loss_disamb = F.binary_cross_entropy_with_logits(outputs.disamb, disamb)

        loss_disamb_obj = F.binary_cross_entropy_with_logits(outputs.disamb_objs, disamb_objects, reduce=False, weight=(1 - object_masks))
        loss_disamb_obj = (loss_disamb_obj * disamb.unsqueeze(-1)).mean()

        loss_act = F.cross_entropy(outputs.acts, acts)

        loss_request_slot = F.binary_cross_entropy_with_logits(outputs.request_slot, request_slot, reduce=False)
        loss_request_slot = (loss_request_slot * request_slot_exist).mean()

        loss_objects = F.binary_cross_entropy_with_logits(outputs.objects, objects, reduce=False, weight=(1 - object_masks))
        loss_objects = (loss_objects * object_exist.unsqueeze(-1)).mean()

        loss_slot_query = F.binary_cross_entropy_with_logits(outputs.slot_query, slot_query)

        loss = loss_disamb + loss_disamb_obj + \
            loss_act + loss_request_slot + loss_objects + loss_slot_query

        acc_disamb = (torch.eq(outputs.disamb > 0, disamb > 0).float()).mean()
        acc_disamb_obj = (torch.eq(outputs.disamb_objs > 0, disamb_objects > 0).float() * (1 - object_masks)).sum() / (1 - object_masks).sum()

        acc_act = (torch.eq(outputs.acts.argmax(-1), acts).float()).mean()

        acc_request_slot_exist = (torch.eq(outputs.request_slot_exist > 0, request_slot_exist > 0).float()).mean()
        acc_request_slot = (torch.eq(outputs.request_slot > 0, request_slot > 0).float()).mean()

        acc_object_exist = (torch.eq(outputs.object_exist > 0, object_exist > 0).float()).mean()
        acc_objects = (torch.eq(outputs.objects > 0, objects > 0).float() * (1 - object_masks)).sum() / (1 - object_masks).sum()

        acc_slot_query = (torch.eq(outputs.slot_query > 0, slot_query > 0).float()).mean()

        self.log("val_loss", loss.item(), prog_bar=True)
        self.log("val_loss_disamb", loss_disamb.item())
        self.log("val_loss_disamb_obj", loss_disamb_obj.item())
        self.log("val_loss_act", loss_act.item())
        self.log("val_loss_request_slot", loss_request_slot.item())
        self.log("val_loss_objects", loss_objects.item())
        self.log("val_loss_slot_query", loss_slot_query.item())

        self.log("val_acc_disamb", acc_disamb.item())
        self.log("val_acc_disamb_obj", acc_disamb_obj.item())

        self.log("val_acc_act", acc_act.item())

        self.log("val_acc_request_slot_exist", acc_request_slot_exist.item())
        self.log("val_acc_request_slot", acc_request_slot.item())

        self.log("val_acc_object_exist", acc_object_exist.item())
        self.log("val_acc_objects", acc_objects.item())

        self.log("val_acc_slot_query", acc_slot_query.item())

    def configure_optimizers(self):
        low_lr_parameters = [
            self.model.bert
        ]

        high_lr_parameters = [
            self.model.context_proj,
            self.model.object_feat,
            self.model.object_proj,
            self.model.disamb_classifier,
            self.model.disamb_head,
            self.model.act_classifier,
            self.model.request_slot_classifier,
            self.model.objects_head,
            self.model.slot_query,
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
