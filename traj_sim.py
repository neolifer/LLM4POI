import io
import os
import copy
import json
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import random
import argparse
import sys
import pickle as pkl
import heapq

import torch
import torch.nn as nn
import transformers
from torch.utils.data import Dataset
from transformers import Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig
from llama_attn_replace_sft import replace_llama_attn
from gptneox_attn_replace import replace_gpt_neox_attn
from peft import LoraConfig, get_peft_model
from torch.distributed import barrier
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="EleutherAI/pythia-1.4b-deduped")
    model_type: Optional[str] = field(default="llama")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192 * 4,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    use_flash_attn: bool = field(
        default=True,
        metadata={"help": "Whether use flash attention for training."},
    )
    use_full_attn: bool = field(
        default=False,
        metadata={"help": "Whether to use plain, full-attention for training."},
    )
    low_rank_training: bool = field(
        default=True,
        metadata={"help": "Whether use low rank adaptation for training."},
    )
    trainable_params: str = field(
        default="embed,norm",
        metadata={"help": "Additional trainable parameters except LoRA weights, if low rank training."},
    )


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size during inference')
    parser.add_argument('--base_model', type=str, default="/data1/pretrained-models/llama-7b-hf")
    parser.add_argument('--cache_dir', type=str, default="./cache")
    parser.add_argument('--seq_len', type=int, default=32768, help='context length during evaluation')
    parser.add_argument('--context_size', type=int, default=32768, help='context size during fine-tuning')
    parser.add_argument('--peft_model', type=str, default=None, help='')
    parser.add_argument('--flash_attn', type=bool, default=True, help='')
    parser.add_argument('--data_path', type=str, default="./test.bin", help='')
    parser.add_argument('--output_dir', type=str, default="/g/data/hn98/peibo/next-poi/outputmodels/finetune-31/",
                        help='')
    parser.add_argument('--dataset_name', type=str, default="nyc",
                        help='')
    args = parser.parse_args()
    return args


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


def get_as_batch(data, seq_length, batch_size, device='cpu', sliding_window=256):
    all_ix = list(range(0, len(data) - seq_length, sliding_window))
    all_ix.pop()

    for idx in range(0, len(all_ix), batch_size):
        ix = all_ix[idx:idx + batch_size]
        assert all([idx + seq_length + 1 <= len(data) for idx in ix])
        x = torch.stack([torch.from_numpy((data[i:i + seq_length]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + seq_length]).astype(np.int64)) for i in ix])
        if device != 'cpu':
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        yield x, y


def iceildiv(x, y):
    return (x + y - 1) // y


def compute_features(hidden, attention):
    averaged_attention = attention.mean(dim=1)
    weighted_hidden_states = torch.zeros_like(hidden)
    batch_size, sequence_length, hidden_size = hidden.shape
    for i in range(batch_size):
        # For each example, perform a weighted sum of hidden states
        # based on the attention weights
        for j in range(sequence_length):
            weighted_hidden_states[i, j, :] = torch.matmul(
                averaged_attention[i, j, :],
                hidden[i, :, :]
            )
    weighted_hidden_states = weighted_hidden_states.mean(axis=[0, 1])
    return weighted_hidden_states


def main(args):
    device = "cuda:0"
    seed = 2
    torch.cuda.set_device(device)

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    model_path = '/g/data/hn98/models/llama2/llama-2-7b-longlora-32k-ft/'
    output_dir = args.output_dir
    print("data path", args.data_path)
    print("base model", model_path)
    print("peft model", output_dir)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=32768,
        padding_side="right",
        use_fast=True,
    )

    # print(tokenizer('6', return_tensors="pt").to(device))
    # print(tokenizer.decode([    29946]))
    # sys.exit()
    # if args.flash_attn:
    #     replace_llama_attn(inference=True)

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_path,
        cache_dir=None,
        output_hidden_states=True,
        output_attentions=True,
        _flash_attn_2_enabled=True
    )

    context_size = args.context_size if args.context_size > 0 else args.seq_len
    orig_ctx_len = getattr(config, "max_position_embeddings", None)  # this value should be 4096 for LLaMA2 models
    if orig_ctx_len and context_size > orig_ctx_len:
        scaling_factor = float(math.ceil(context_size / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='auto',
        config=config,
        cache_dir=None,
        torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
    )
    model.resize_token_embeddings(32001)

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    targets = ["q_proj", "k_proj", "v_proj", "o_proj"]

    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=targets,
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.eval()

    data_path = f'datasets/processed/{args.dataset_name}/'

    def compute_fea(train=True):
        key_query_traj = {}
        if train:
            output = 'train'
        else:
            output = 'test'
        if train:
            list_data_dict = jload(data_path + f'{output}_kq_pairs.json')
        else:
            list_data_dict = jload(data_path + f'{output}_kq_pairs.json')
        for e in tqdm(list_data_dict, desc="Processing lines", total=len(list_data_dict)):
            try:
                key = tokenizer(e['key'], return_tensors="pt").to(device)
                key = model(**key)
                key = compute_features(key.hidden_states[-1], key.attentions[-1]).cpu().detach()
                torch.cuda.empty_cache()
                query = tokenizer(e['query'], return_tensors="pt").to('cuda:1')
                query = model(**query)
                query = compute_features(query.hidden_states[-1], query.attentions[-1]).cpu().detach()
                torch.cuda.empty_cache()
                key_query_traj[e['traj_id']] = {'key': key, 'query': query, 'start_time': e['start_time'], 'end_time':e['end_time']}
            except Exception as ex:
                print(f"An error occurred: {ex}")  # Log the exception
                continue

        with open(data_path + f'{output}_kqt.pkl', 'wb') as fp:
            pkl.dump(key_query_traj, fp)

    compute_fea(True)
    compute_fea(False)

    def compute_sim(train=True):
        if train:
            with open(data_path + 'train_kqt.pkl', 'rb') as fp:
                key_query_traj_train = pkl.load(fp)
            with open(data_path + 'train_kqt.pkl', 'rb') as fp:
                key_query_traj = pkl.load(fp)
        else:
            with open(data_path + 'train_kqt.pkl', 'rb') as fp:
                key_query_traj_train = pkl.load(fp)
            with open(data_path + 'test_kqt.pkl', 'rb') as fp:
                key_query_traj = pkl.load(fp)
        # Assuming key_query_traj is already populated with PyTorch tensors
        results = {}
        gpus = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]

        # Function to compute similarity for a subset of data on a specific GPU
        def compute_similarity_on_gpu(subset, other_subset, gpu):
            local_results = {}

            # Stack all query tensors for batch processing
            query_tensors = [other_element['query'].to(gpu).reshape(1, -1) for other_element in other_subset.values()]
            stacked_queries = torch.vstack(query_tensors)

            # Convert start_times and end_times from strings to floats
            start_times = torch.tensor([float(element['start_time']) for element in subset.values()], device=gpu)
            end_times = torch.tensor([float(other_element['end_time']) for other_element in other_subset.values()],
                                     device=gpu)

            # Pre-compute a matrix for time condition checks
            time_condition_matrix = start_times[:, None] > end_times

            for traj_id, element in tqdm(subset.items(), desc="Computing Similarities"):
                key_tensor = element['key'].to(gpu).reshape(1, -1)

                # Filter queries based on time condition
                valid_indices = time_condition_matrix[list(subset.keys()).index(traj_id)]
                filtered_queries = stacked_queries[valid_indices]

                if len(filtered_queries) == 0:
                    continue
                # Also filter the trajectory IDs
                filtered_traj_ids = [traj_id for traj_id, valid in zip(list(other_subset.keys()), valid_indices.cpu().numpy()) if valid]
        
                # Compute similarities only for the filtered queries
                batch_similarities = F.cosine_similarity(key_tensor, filtered_queries)

                # Extract top 35 similarities
                top_k = min(len(filtered_queries), 35)

                # Extract top similarities
                if top_k > 0:
                    _, top_indices = torch.topk(batch_similarities, k=top_k)
                    top_queries_traj_ids = [filtered_traj_ids[idx] for idx in top_indices.cpu().numpy()]
                    local_results[traj_id] = top_queries_traj_ids
                else:
                    local_results[traj_id] = []
            return local_results
        # Divide the data among GPUs
        data_subsets = {gpu: {} for gpu in gpus}
        for i, (traj_id, data) in enumerate(key_query_traj.items()):
            gpu = gpus[i % len(gpus)]
            data_subsets[gpu][traj_id] = data

        # Initialize a global progress bar
        total_tasks = len(key_query_traj)
        progress_bar = tqdm(total=total_tasks, desc="Overall Progress")

        # Compute similarities in parallel on multiple GPUs
        with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
            futures = []
            for gpu in gpus:
                future = executor.submit(compute_similarity_on_gpu, data_subsets[gpu], key_query_traj_train, gpu)
                futures.append(future)
                # Update the progress bar immediately after task submission
                progress_bar.update(1)

            # Retrieve results from completed futures
            for future in futures:
                local_results = future.result()
                results.update(local_results)

        progress_bar.close()
        if train:
            with open(data_path + 'train_key_top200.json', 'w') as fp:
                json.dump(results, fp)
        else:
            with open(data_path + 'test_key_top200.json', 'w') as fp:
                json.dump(results, fp)
    compute_sim(True)
    compute_sim(False)

if __name__ == "__main__":
    args = parse_config()
    main(args)
