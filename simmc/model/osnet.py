from collections import namedtuple

import torch
import torch.nn as nn

from transformers import AlbertModel

from simmc.data.preprocess import OBJECT_FEATURE_SIZE
import simmc.data.labels as L


OSNetOutput = namedtuple("OSNetOutput",
    ["context_proj", "object_proj", "disamb", "disamb_proj", "acts", "is_request", "slots"])

# Object Sentence network
class OSNet(nn.Module):
    def __init__(self):
        super(OSNet, self).__init__()

        self.bert = AlbertModel.from_pretrained("albert-base-v2")
        for param in self.bert.parameters():
            param.requires_grad = False

        self.projection_dim = 512

        self.context_proj = nn.Linear(768, self.projection_dim)

        self.object_feat = nn.Sequential(
            nn.Linear(OBJECT_FEATURE_SIZE, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256)
        )

        self.object_proj = nn.Linear(256, self.projection_dim)

        self.disamb_classifier = nn.Linear(768, 1)
        self.disamb_proj = nn.Linear(256, self.projection_dim)

        self.act_classifier = nn.Linear(768, len(L.ACTION))
        self.is_req_classifier = nn.Linear(768, 1)
        self.slot_classifier = nn.Linear(768, len(L.SLOT_KEY))


    def forward(self, context_inputs, objects):
        context_feat = self.bert(**context_inputs).last_hidden_state[:, 0, :]
        object_feat = self.object_feat(objects)

        context_proj = self.context_proj(context_feat)
        object_proj = self.object_proj(object_feat)

        disamb = self.disamb_classifier(context_feat)
        disamb_proj = self.disamb_proj(object_feat)

        act = self.act_classifier(context_feat)
        is_req = self.is_req_classifier(context_feat)
        slot = self.slot_classifier(context_feat)

        return OSNetOutput(
            context_proj=context_proj,
            object_proj=object_proj,
            disamb=disamb,
            disamb_proj=disamb_proj,
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

    net = OSNet()

    tokenizer = AlbertTokenizerFast.from_pretrained("albert-base-v2")

    dataset = OSDataset("/data/simmc2/train_dials.pkl")
    data = choice(dataset) 
    print(data)

    context_inputs = tokenizer([data["context"]], padding=True, truncation=True, return_tensors="pt")

    context, objects = net(context_inputs, data["objects"].unsqueeze(0))
    print("context shape:", context.shape)
    print("objects shape:", objects.shape)
    print("dot sim:", calc_object_similarity(context, objects, data["object_masks"].unsqueeze(0)))
