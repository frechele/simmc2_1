import torch
import torch.nn as nn

from transformers import AlbertModel

from simmc.data.preprocess import OBJECT_FEATURE_SIZE


# Object Sentence network
class OSNet(nn.Module):
    def __init__(self):
        super(OSNet, self).__init__()

        self.bert = AlbertModel.from_pretrained("albert-base-v2")
        for param in self.bert.parameters():
            param.requires_grad = False

        self.projection_dim = 512

        self.context_proj = nn.Linear(768, self.projection_dim)

        self.object_proj = nn.Sequential(
            nn.Linear(OBJECT_FEATURE_SIZE, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.projection_dim)
        )

    def forward(self, context_inputs, objects):
        context_feat = self.bert(**context_inputs).last_hidden_state[:, 0, :]

        context_proj = self.context_proj(context_feat)
        object_proj = self.object_proj(objects)

        return context_proj, object_proj


def calc_object_similarity(context: torch.Tensor, objects: torch.Tensor, object_mask: torch.Tensor):
    sim = torch.einsum("bc,boc->bo", context, objects)
    return sim.masked_fill(object_mask, -1e5)


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
