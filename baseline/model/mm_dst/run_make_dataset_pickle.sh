#! /bin/bash

if [[ $# -lt 1 ]]
then
    PATH_DATA_DIR=$(realpath ../../data)
else
    PATH_DATA_DIR=$(realpath "$1")
fi

# Train split
python3 -m gpt2_dst.scripts.make_dataset_pickle \
    --predict="${PATH_DATA_DIR}"/simmc2.1_dials_dstc11_train_predict.txt \
    --target="${PATH_DATA_DIR}"/simmc2.1_dials_dstc11_train_target.txt \
    --output="${PATH_DATA_DIR}"/simmc2.1_dials_dstc11_train.pkl \

# Dev split
python3 -m gpt2_dst.scripts.make_dataset_pickle \
    --predict="${PATH_DATA_DIR}"/simmc2.1_dials_dstc11_dev_predict.txt \
    --target="${PATH_DATA_DIR}"/simmc2.1_dials_dstc11_dev_target.txt \
    --output="${PATH_DATA_DIR}"/simmc2.1_dials_dstc11_dev.pkl \

# Devtest split
python3 -m gpt2_dst.scripts.make_dataset_pickle \
    --predict="${PATH_DATA_DIR}"/simmc2.1_dials_dstc11_devtest_predict.txt \
    --target="${PATH_DATA_DIR}"/simmc2.1_dials_dstc11_devtest_target.txt \
    --output="${PATH_DATA_DIR}"/simmc2.1_dials_dstc11_devtest.pkl \
