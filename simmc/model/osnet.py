from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AlbertModel, AlbertTokenizer

from simmc.data.metadata import MetadataDB
import simmc.data.labels as L
from simmc.model.object_enc import ObjectEncoder


BERT_MODEL_NAME = "albert-base-v2"


def create_tokenizer():
    tokenizer = AlbertTokenizer.from_pretrained(BERT_MODEL_NAME)
    tokenizer.add_special_tokens({"additional_special_tokens": [L.USER_UTTR_TOKEN, L.SYSTEM_UTTR_TOKEN]})

    return tokenizer


class CrossAttention(nn.Module):
    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super(CrossAttention, self).__init__()

        self.embed_dim = embed_dim
        self.scale = embed_dim ** -0.5

        self.q_fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self.kv_fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2)
        )

        self.o_fc = nn.Sequential(
            nn.Dropout(dropout, inplace=True),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.Dropout(dropout, inplace=True),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, context: torch.Tensor, object_map: torch.Tensor, object_mask: torch.Tensor) -> torch.Tensor:
        query = self.q_fc(context)
        key, value = self.kv_fc(object_map).split(self.embed_dim, dim=-1)

        x_qk = torch.einsum("bi, boj -> bo", query, key) * self.scale
        x_qk = x_qk.masked_fill(object_mask, -1e4)
        x_qk = torch.softmax(x_qk, dim=-1)

        x_qkv = torch.einsum("bo, boi -> boi", x_qk, value)
        x_qkv = x_qkv + context.unsqueeze(1).expand_as(x_qkv)

        return x_qkv + self.o_fc(x_qkv)


OSNetOutput = namedtuple("OSNetOutput",
    ["disamb", "disamb_objs", "acts", 
    "request_slot_exist", "request_slot",
    "object_exist", "objects", "slot_query"])

# Object Sentence network
class OSNet(nn.Module):
    def __init__(self, db: MetadataDB):
        super(OSNet, self).__init__()

        self.db = db

        self.bert = AlbertModel.from_pretrained(BERT_MODEL_NAME)
        self.bert.resize_token_embeddings(self.bert.config.vocab_size + 2)

        self.projection_dim = 256 

        self.context_proj = nn.Linear(768, self.projection_dim)

        self.object_feat = ObjectEncoder(self.db, self.projection_dim)
        self.object_proj = nn.Sequential(
            CrossAttention(self.projection_dim),
            nn.Mish(inplace=True),
            nn.Linear(self.projection_dim, 1)
        )

        self.disamb_classifier = nn.Linear(self.projection_dim, 1)
        self.disamb_head = CrossAttention(self.projection_dim)

        self.act_classifier = nn.Linear(self.projection_dim, len(L.ACTION))

        self.request_slot_exist = nn.Linear(self.projection_dim, 1)
        self.request_slot_classifier = nn.Linear(self.projection_dim, len(L.SLOT_KEY))

        self.object_exist = nn.Linear(self.projection_dim, 1)
        self.objects_head = nn.Sequential(
            CrossAttention(self.projection_dim),
            nn.Mish(inplace=True),
            nn.Linear(self.projection_dim, 1)
        )

        self.slot_query = nn.Linear(self.projection_dim, len(L.SLOT_KEY))

    def forward(self, context_inputs: torch.Tensor, object_map: torch.Tensor, object_mask: torch.Tensor):
        context_feat = self.bert(**context_inputs).last_hidden_state[:, 0, :]
        context_proj = self.context_proj(context_feat)

        object_feat = self.object_feat(context_proj, object_map)

        disamb = self.disamb_classifier(context_proj)

        disamb_objs = self.disamb_head(context_proj, object_feat, object_mask).squeeze(-1)
        disamb_objs.masked_fill_(object_mask, -1e4)

        act = self.act_classifier(context_proj)

        request_slot_exist = self.request_slot_exist(context_proj)
        request_slot = self.request_slot_classifier(context_proj)

        object_exist = self.object_exist(context_proj)
        objects = self.objects_head(context_proj, object_feat, object_mask).squeeze(-1)
        objects.masked_fill_(object_mask, -1e4)

        slot_query = self.slot_query(context_proj)

        return OSNetOutput(
            disamb=disamb,
            disamb_objs=disamb_objs,
            acts=act,

            request_slot_exist=request_slot_exist,
            request_slot=request_slot,

            object_exist=object_exist,
            objects=objects,
            slot_query=slot_query
        )


def calc_object_similarity(context: torch.Tensor, objects: torch.Tensor):
    return F.cosine_similarity(context.unsqueeze(1).expand_as(objects), objects, dim=-1) * 100


if __name__ == "__main__":
    from simmc.data.os_dataset import OSDataset
    from transformers import AlbertTokenizerFast
    from random import choice
    from torchinfo import summary

    from simmc.data.metadata import MetadataDB

    db = MetadataDB("/data/simmc2/metadata_db.pkl")

    net = OSNet(db)

    tokenizer = AlbertTokenizerFast.from_pretrained("albert-base-v2")

    dataset = OSDataset("/data/simmc2/train_dials.pkl")
    data = choice(dataset) 
    print(data)

    context_inputs = tokenizer([data["context"]], padding=True, truncation=True, return_tensors="pt")

    outputs = net(context_inputs, data["object_map"].unsqueeze(0), data["object_masks"].unsqueeze(0))
    print(outputs)

    summary(net)
