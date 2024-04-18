#!/bin/bash

DATASET_NAME="ca"

torchrun --nproc_per_node=8 supervised-fine-tune-qlora.py  \
--model_name_or_path /g/data/hn98/models/llama2/llama-2-7b-longlora-32k-ft \
--bf16 True \
--output_dir /g/data/hn98/peibo/next-poi/outputmodels/finetune-58       \
--model_max_length 32768 \
--use_flash_attn True \
--data_path /g/data/hn98/peibo/next-poi/dataset/processed/${DATASET_NAME}/train_qa_pairs_kqt.json \
--low_rank_training True \
--num_train_epochs 3  \
--per_device_train_batch_size 1     \
--per_device_eval_batch_size 2     \
--gradient_accumulation_steps 1     \
--evaluation_strategy "no"     \
--save_strategy "steps"     \
--save_steps 1000     \
--save_total_limit 2     \
--learning_rate 2e-5     \
--weight_decay 0.0     \
--warmup_steps 20     \
--lr_scheduler_type "constant_with_warmup"     \
--logging_steps 1     \
--deepspeed "ds_configs/stage2.json" \
--tf32 True

python eval_next_poi.py --dataset_name ${DATASET_NAME} --output_dir /g/data/hn98/peibo/next-poi/outputmodels/finetune-58 --test_file "test_qa_pairs_kqt.txt"
#
#
