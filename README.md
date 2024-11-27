# Large Language Models for Next Point-of-Interest Recommendation
[![License: APACHE-2.0](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://www.apache.org/licenses/LICENSE-2.0)
[![Venue:SIGIR 2024](https://img.shields.io/badge/Venue-SIGIR2024-orange)](https://sigir-2024.github.io/index.html)

This repository includes the implementation of paper "[Large Language Models for Next Point-of-Interest Recommendation](https://arxiv.org/pdf/2404.17591)".
# Install
1. Clone this repository to your local machine.
2. Install the enviroment by running
```
conda env create -f environment.yml
```
Alternatively, you can download the conda environment in linux directly with this [google drive link](https://drive.google.com/file/d/1SKKSwjdEapQh5WOEpv8XkLZTTkhlKDg6/view?usp=sharing).
Then try:

```
mkdir -p llm4poi
tar -xzf "venv.tar.gz" -C "llm4poi"
conda activate llm4poi
```

3. Download the model from (https://huggingface.co/Yukang/Llama-2-7b-longlora-32k-ft)
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
Alternatively, you may directly use the files from traj_sim.zip.

run ```python preprocessing/to_nextpoi_qkt.py --dataset_name {dataset_name}```


# Main Performance
## train
run
```
torchrun --nproc_per_node=8 supervised-fine-tune-qlora.py  \
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
python eval_next_poi.py --model_path {your_model_path}--dataset_name {DATASET_NAME} --output_dir {your_finetuned_model} --test_file "test_qa_pairs_kqt.txt"
```
## Acknowledgement
This code is developed based on [STHGCN](https://github.com/ant-research/Spatio-Temporal-Hypergraph-Model) and [LongLoRA](https://github.com/dvlab-research/LongLoRA?tab=readme-ov-file).
## Citation
If you find our work useful, please consider cite our paper with following:
```
@inproceedings{li-2024-large,
author = {Li, Peibo and de Rijke, Maarten and Xue, Hao and Ao, Shuang and Song, Yang and Salim, Flora D.},
booktitle = {SIGIR 2024: 47th international ACM SIGIR Conference on Research and Development in Information Retrieval},
date-added = {2024-03-26 23:47:40 +0000},
date-modified = {2024-03-26 23:48:47 +0000},
month = {July},
publisher = {ACM},
title = {Large Language Models for Next Point-of-Interest Recommendation},
year = {2024}}
```
