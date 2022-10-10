#! /bin/bash

DATA_ROOT=$1

# public dataset
python3 -m simmc.script.build_scene_info \
    --fashion_metadata ${DATA_ROOT}/fashion_prefab_metadata_all.json \
    --furniture_metadata ${DATA_ROOT}/furniture_prefab_metadata_all.json \
    --scene_root ${DATA_ROOT}/public \
    --output ${DATA_ROOT}/public_scene_info.pkl
