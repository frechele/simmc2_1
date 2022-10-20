import numpy as np

from simmc.data.metadata import MetadataDB
import simmc.data.labels as L
from simmc.data.labels import label_to_onehot, labels_to_vector


def metadata_to_feat(metadata, db: MetadataDB):
    domain = metadata["domain"]

    if domain == "fashion":
        return fashion_metadata_to_feat(metadata, db)
    elif domain == "furniture":
        return furniture_metadata_to_feat(metadata, db)

    raise ValueError(f"Unknown domain: {domain}")

def fashion_metadata_to_feat(metadata, db: MetadataDB):
    slot_idxes = []
    slot_values = []

    for k in ["brand", "type", "assetType", "pattern"]:
        slot_idxes.append(db.get_key_idx(k))
        slot_values.append(db.get_idx(k, metadata.get(k, None)))

    return np.array(slot_idxes), np.array(slot_values)


def furniture_metadata_to_feat(metadata, db: MetadataDB):
    slot_idexs = []
    slot_values = []

    for k in ["brand", "type", "materials", db.pad_str]:
        slot_idexs.append(db.get_key_idx(k))
        slot_values.append(db.get_idx(k, metadata.get(k, None)))

    return np.array(slot_idexs), np.array(slot_values)
