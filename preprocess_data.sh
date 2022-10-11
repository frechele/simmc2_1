#! /bin/bash

DATA_ROOT=$1

# public dataset
echo "build public scene info"
python3 -m simmc.script.build_scene_info \
    --fashion_metadata ${DATA_ROOT}/fashion_prefab_metadata_all.json \
    --furniture_metadata ${DATA_ROOT}/furniture_prefab_metadata_all.json \
    --scene_root ${DATA_ROOT}/public \
    --output ${DATA_ROOT}/public_scene_info.pkl

echo "build train dialog dataset"
python3 -m simmc.script.build_pretrain_dataset \
    --dialog ${DATA_ROOT}/simmc2.1_dials_dstc11_train.json \
    --scene ${DATA_ROOT}/public_scene_info.pkl \
    --output ${DATA_ROOT}/train_dials.pkl

echo "build dev dialog dataset"
python3 -m simmc.script.build_pretrain_dataset \
    --dialog ${DATA_ROOT}/simmc2.1_dials_dstc11_dev.json \
    --scene ${DATA_ROOT}/public_scene_info.pkl \
    --output ${DATA_ROOT}/dev_dials.pkl

echo "build devtest dialog dataset"
python3 -m simmc.script.build_pretrain_dataset \
    --dialog ${DATA_ROOT}/simmc2.1_dials_dstc11_devtest.json \
    --scene ${DATA_ROOT}/public_scene_info.pkl \
    --output ${DATA_ROOT}/devtest_dials.pkl

# test dataset (TODO)
