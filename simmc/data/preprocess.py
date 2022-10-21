import numpy as np

from simmc.data.metadata import MetadataDB


FEATURE_LIST = ["brand", "type", "assetType", "pattern", "materials"]


def label_to_onehot(value, mapping_table):
    return np.eye(len(mapping_table))[mapping_table[value]]

def labels_to_vector(values, mapping_table):
    return np.sum([label_to_onehot(value, mapping_table) for value in values], axis=0)


def metadata_to_feat(metadata, db: MetadataDB):
    slot_values = []

    for k in FEATURE_LIST:
        slot_values.append(db.get_idx(k, metadata.get(k, db.pad_str)))

    return np.array(slot_values)
