#! /bin/bash

DATA_PATH="/data/simmc2"

python train_model.py \
	--train_file "${DATA_PATH}/ambiguous_candidates/simmc2.1_ambiguous_candidates_dstc11_train.json" \
	--dev_file "${DATA_PATH}/ambiguous_candidates/simmc2.1_ambiguous_candidates_dstc11_dev.json" \
	--devtest_file "${DATA_PATH}/ambiguous_candidates/simmc2.1_ambiguous_candidates_dstc11_devtest.json" \
	--teststd_file "${DATA_PATH}/ambiguous_candidates/simmc2.1_ambiguous_candidates_dstc11_devtest.json" \
	--result_save_path "${DATA_PATH}/ambiguous_candidates/results/" \
	--visual_feature_path "${DATA_PATH}/visual_features_resnet50_simmc2.1.pt" \
	--visual_feature_size 516 \
	--backbone bert --use_gpu --num_epochs 10 --batch_size 16 --max_turns 2

