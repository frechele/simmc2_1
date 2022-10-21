import torch
import torch.nn as nn

from simmc.data.metadata import MetadataDB
from simmc.data.preprocess import FEATURE_LIST


class ObjectEncoder(nn.Module):
    def __init__(self, db: MetadataDB, embed_size: int):
        super(ObjectEncoder, self).__init__()

        self.db = db
        self.embed_size = embed_size

        self.slot_attn = nn.Sequential(
            nn.Linear(embed_size, embed_size // 16),
            nn.Mish(inplace=True),
            nn.Linear(embed_size // 16, len(FEATURE_LIST)),
        )

        self.encoders = nn.ModuleList([
            self._build_encoder(k) for k in FEATURE_LIST
        ])


    def forward(self, context: torch.Tensor, object_map: torch.Tensor) -> torch.Tensor:
        embeddings = []
        for i, encoder in enumerate(self.encoders):
            values = object_map[..., i]
            embeddings.append(encoder(values).unsqueeze(1))

        embeddings = torch.concat(embeddings, dim=1) 

        slot_weight = self.slot_attn(context)
        slot_weight = torch.softmax(slot_weight, dim=-1)

        return torch.einsum("bsoi, bs -> boi", embeddings, slot_weight)


    def _build_encoder(self, key: str):
        cardinality = self.db.get_cardinality(key)
        return nn.Sequential(
            nn.Embedding(cardinality, self.embed_size, self.db.pad_idx),
            nn.LayerNorm(self.embed_size),
        )
