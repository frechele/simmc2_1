from collections import namedtuple

import torch
import torch.nn as nn

from transformers import AlbertModel

from simmc.data.preprocess import OBJECT_FEATURE_SIZE
import simmc.data.labels as L


class OSBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super(OSBlock, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dim_per_head = embed_dim // num_heads

        self.w_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w_kv = nn.Linear(embed_dim, embed_dim * 2, bias=False)
        self.scale = embed_dim ** -0.5

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, context: torch.Tensor, objects: torch.Tensor, object_mask: torch.Tensor):
        # context: [batch_size, embed_dim]
        # objects: [batch_size, object_num, embed_dim]
        # object_mask: [batch_size, object_num]
        batch_size = context.size(0)
        object_size = objects.size(1)

        query = self.w_q(context)
        key, value = torch.split(self.w_kv(objects), self.embed_dim, dim=-1)

        query = query.view(batch_size, self.num_heads, self.dim_per_head)
        key = key.view(batch_size, -1, self.num_heads, self.dim_per_head)
        value = value.view(batch_size, -1, self.num_heads, self.dim_per_head)

        z = torch.einsum("bhi, bohi -> boh", query, key) * self.scale
        masking_value = -torch.finfo(z.dtype).max
        object_mask = object_mask.unsqueeze(-1)
        object_mask = object_mask.expand(-1, -1, self.num_heads)
        z.masked_fill_(object_mask, masking_value)

        attn = torch.softmax(z, dim=-1)

        out = torch.einsum("boh, bohi -> bohi", attn, value)
        out = out.view(batch_size, object_size, -1)

        out = self.ffn(out) + objects

        return out


class OSTransformer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super(OSTransformer, self).__init__()

        self.blocks = nn.ModuleList([
            OSBlock(embed_dim, num_heads, dropout=0.2) for _ in range(3)
            ])

    def forward(self, context: torch.Tensor, objects: torch.Tensor, object_mask: torch.Tensor):
        path = objects 
        attn = None

        for block in self.blocks:
            path = block(context, path, object_mask)

        return path


OSNetOutput = namedtuple("OSNetOutput",
    ["objects", "disamb", "disamb_objs", "acts", "is_request", "slots"])

# Object Sentence network
class OSNet(nn.Module):
    def __init__(self):
        super(OSNet, self).__init__()

        self.bert = AlbertModel.from_pretrained("albert-base-v2")
        for p in self.bert.parameters():
            p.requires_grad = False

        self.projection_dim = 256 

        self.context_proj = nn.Sequential(
            nn.Linear(768, 512),
            nn.Mish(inplace=True),
            nn.Linear(512, self.projection_dim)
        )

        self.object_feat = nn.Sequential(
            nn.Linear(OBJECT_FEATURE_SIZE, 128),
            nn.Mish(inplace=True),
            nn.Linear(128, 256),
            nn.Mish(inplace=True)
        )

        self.os_trans = OSTransformer(self.projection_dim, 8)

        self.object_head = nn.Linear(self.projection_dim, self.projection_dim)

        self.disamb_classifier = nn.Linear(self.projection_dim, 1)
        self.disamb_head = nn.Linear(self.projection_dim, self.projection_dim)

        self.act_classifier = nn.Linear(self.projection_dim, len(L.ACTION))
        self.is_req_classifier = nn.Linear(self.projection_dim, 1)
        self.slot_classifier = nn.Linear(self.projection_dim, len(L.SLOT_KEY))


    def forward(self, context_inputs: torch.Tensor, objects: torch.Tensor, object_mask: torch.Tensor):
        context_feat = self.bert(**context_inputs).last_hidden_state[:, 0, :]
        object_feat = self.object_feat(objects)

        context_proj = self.context_proj(context_feat)

        object_proj = self.os_trans(context_proj, object_feat, object_mask)

        objects = self.object_head(object_proj)
        objects = calc_object_similarity(context_proj, objects)
        objects.masked_fill_(object_mask, -1e4)

        disamb = self.disamb_classifier(context_proj)

        disamb_objs = self.disamb_head(object_proj)
        disamb_objs = calc_object_similarity(context_proj, disamb_objs)
        disamb_objs.masked_fill_(object_mask, -1e4)

        act = self.act_classifier(context_proj)
        is_req = self.is_req_classifier(context_proj)
        slot = self.slot_classifier(context_proj)

        return OSNetOutput(
            objects=objects,
            disamb=disamb,
            disamb_objs=disamb_objs,
            acts=act,
            is_request=is_req,
            slots=slot
        )


def calc_object_similarity(context: torch.Tensor, objects: torch.Tensor):
    sim = torch.einsum("bc,boc->bo", context, objects)
    return sim


if __name__ == "__main__":
    from simmc.data.os_dataset import OSDataset
    from transformers import AlbertTokenizerFast
    from random import choice
    from torchinfo import summary

    net = OSNet()

    tokenizer = AlbertTokenizerFast.from_pretrained("albert-base-v2")

    dataset = OSDataset("/data/simmc2/train_dials.pkl")
    data = choice(dataset) 
    print(data)

    context_inputs = tokenizer([data["context"]], padding=True, truncation=True, return_tensors="pt")

    outputs = net(context_inputs, data["objects"].unsqueeze(0), data["object_masks"].unsqueeze(0))
    print(outputs)

    summary(net)
