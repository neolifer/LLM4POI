# LLM4POI
This repository includes the implementation of "Large Language Models for Next Point-of-Interest Recommendation".
# Install
Install the enviroment by running
```
conda env create -f environment.yml
```
Download the model from (https://huggingface.co/Yukang/Llama-2-7b-longlora-32k-ft)
# Dataset
Download the datasets raw data from [datasets](https://www.dropbox.com/scl/fi/teo5pn8t296joue5c8pim/datasets.zip?rlkey=xvcgtdd9vlycep3nw3k17lfae&st=qd21069y&dl=0).
* Unzip datasets.zip to ./datasets
* Unzip datasets/nyc/raw.zip to datasets/nyc.
* Unzip datasets/tky/raw.zip to datasets/tky.
* Unzip datasets/ca/raw.zip to datasets/ca.
* run ```python preprocesssing/generate_ca_raw.py --dataset_name {dataset_name}```

# Preprocess
run ```python preprocessing/run.py```

run ```python preprocessing/traj_qk.py```

run ```python traj_sim --dataset_name {dataset_name} --model_path {your_model_path}```


# Main Performance
## train
run
```torchrun --nproc_per_node=8 supervised-fine-tune-qlora.py  \
--model_name_or_path {your_model_path} \
--bf16 True \
--output_dir {your_output_path}\
--model_max_length 32768 \
--use_flash_attn True \
--data_path datasets/processed/{DATASET_NAME}/train_qa_pairs_kqt.json \
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
```

## test
run
```
python eval_next_poi.py --dataset_name {DATASET_NAME} --output_dir {your_finetuned_model} --test_file "test_qa_pairs_kqt.txt"
```

