import torch
import torch.nn as nn

from simmc.data.metadata import MetadataDB
from simmc.data.preprocess import FEATURE_LIST


class ObjectEncoder(nn.Module):
    def __init__(self, db: MetadataDB, embed_size: int):
        super(ObjectEncoder, self).__init__()

        self.db = db
        self.embed_size = embed_size

        self.encoders = nn.ModuleList([
            self._build_encoder(k) for k in FEATURE_LIST
        ])


    def forward(self, object_map: torch.Tensor, slot_weight: torch.Tensor) -> torch.Tensor:
        embeddings = []
        for i, encoder in enumerate(self.encoders):
            values = object_map[..., i]
            embeddings.append(encoder(values).unsqueeze(1))

        embeddings = torch.concat(embeddings, dim=1) 

        return torch.einsum("bsoi, bs -> boi", embeddings, slot_weight)


    def _build_encoder(self, key: str):
        cardinality = self.db.get_cardinality(key)
        return nn.Sequential(
            nn.Embedding(cardinality, self.embed_size, self.db.pad_idx),
            nn.LayerNorm(self.embed_size),
        )


if __name__ == "__main__":
    from simmc.data.os_dataset import OSDataset
    from random import choice
    from torchinfo import summary

    db = MetadataDB("/data/simmc2/metadata_db.pkl")
    dataset = OSDataset("/data/simmc2/train_dials.pkl")
    data = choice(dataset)

    net = ObjectEncoder(db, 256)
    summary(net)

    object_map = data["object_map"].unsqueeze(0)
    slot_weights = torch.randn(1, len(FEATURE_LIST))
    outputs = net(object_map, slot_weights)
    print(outputs.shape)
