import numpy as np

from simmc.data.metadata import MetadataDB
import simmc.data.labels as L
from simmc.data.labels import label_to_onehot, labels_to_vector


FEATURE_LIST = ["brand", "type", "assetType", "materials"]


def metadata_to_feat(metadata, db: MetadataDB):
    domain = metadata["domain"]

    if domain == "fashion":
        return fashion_metadata_to_feat(metadata, db)
    elif domain == "furniture":
        return furniture_metadata_to_feat(metadata, db)

    raise ValueError(f"Unknown domain: {domain}")

def fashion_metadata_to_feat(metadata, db: MetadataDB):
    slot_values = []

    fashion_feature_list = FEATURE_LIST[:]
    fashion_feature_list[-1] = "pattern"

    for k in fashion_feature_list:
        slot_values.append(db.get_idx(k, metadata.get(k, db.pad_str)))

    return np.array(slot_values)


def furniture_metadata_to_feat(metadata, db: MetadataDB):
    slot_values = []

    for k in FEATURE_LIST:
        slot_values.append(db.get_idx(k, metadata.get(k, db.pad_str)))

    return np.array(slot_values)
