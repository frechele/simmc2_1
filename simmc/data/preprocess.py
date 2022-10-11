import numpy as np

import simmc.data.labels as L
from simmc.data.labels import label_to_onehot, labels_to_vector


def metadata_to_vec(metadata):
    domain = metadata["domain"]

    if domain == "fashion":
        return fashion_metadata_to_vec(metadata)
    elif domain == "furniture":
        return furniture_metadata_to_vec(metadata)

    raise ValueError(f"Unknown domain: {domain}")


def fashion_metadata_to_vec(metadata):
    obj_type = label_to_onehot(metadata["type"], L.TYPE_MAPPING_TABLE)
    asset_type = label_to_onehot(metadata["assetType"], L.ASSET_TYPE_MAPPING_TABLE)
    brand = label_to_onehot(metadata["brand"], L.BRAND_MAPPING_TABLE)
    material = label_to_onehot(metadata["pattern"], L.MATERIAL_MAPPING_TABLE)
    color = labels_to_vector(metadata["color"], L.COLOR_MAPPING_TABLE)

    return np.concatenate([obj_type, asset_type, brand, material, color])


def furniture_metadata_to_vec(metadata):
    obj_type = label_to_onehot(metadata["type"], L.TYPE_MAPPING_TABLE)
    asset_type = np.zeros(len(L.ASSET_TYPE_MAPPING_TABLE))
    brand = label_to_onehot(metadata["brand"], L.BRAND_MAPPING_TABLE)
    material = label_to_onehot(metadata["materials"], L.MATERIAL_MAPPING_TABLE)
    color = labels_to_vector(metadata["color"], L.COLOR_MAPPING_TABLE)

    return np.concatenate([obj_type, asset_type, brand, material, color])
