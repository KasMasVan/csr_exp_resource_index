import argparse
import csv
import logging
import os
import random
import sys
from tqdm import tqdm

import numpy as np
import torch
from transformers import(
    AutoTokenizer, 
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)
from datasets import Dataset

# import data.py, which is located in the same directory

from .data import(
    copa_loader,
    cqa_loader,
    obqa_loader,
    piqa_loader,
    qasc_loader,
    siqa_loader,
    winogrande_loader,
)

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser("Inference on multiple choice benchmarks")
    parser.add_argument(
        "--seed", 
        type=int, 
        default=0,
        help="Random seed for reproducibility.",
        )
    parser.add_argument(
        "--model_family",
        type=str,
        choices=["GPT2", "T5", "FLAN-T5", "Pythia"],
        default=None,
        required=True,
        help="The moddel family, as checkpoints under the same model family use same codes to download.",
        )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        required=True,
        help="The checkpoint name under a model family, e.g. gpt2, gpt2-medium, gpt2-large, gpt2-xl.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        # choices=["copa", "cqa", "winogrande"],
        default=None,
        required=True,
        help="The datasets to inference on. Pass multiple datasets separate by space",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--multiple_choice_prompt",
        type=str,
        default = None,
        help = "The multiple choice prompt."
    )

    args = parser.parse_args()
    return args

def load_data(args):
    if args.dataset == "copa":
        ending_names = ['hypothesis0', 'hypothesis1']
        header_name = "premise"
        file_path = os.path.join("../data", args.dataset, "copa-dev.xml")
        loader = copa_loader
    elif args.dataset == "cqa":
        ending_names = ['hypothesis0', 'hypothesis1', 'hypothesis2', 'hypothesis3', 'hypothesis4']
        header_name = "premise"
        file_path = os.path.join("../data", args.dataset, "dev.jsonl")
        loader = cqa_loader
    elif args.dataset == "obqa":
        ending_names = ['hypothesis0', 'hypothesis1', 'hypothesis2', 'hypothesis3']
        header_name = "premise"
        file_path = os.path.join("../data", args.dataset, "dev.jsonl")
        loader = obqa_loader
    elif args.dataset == "piqa":
        ending_names = ['hypothesis0', 'hypothesis1']
        header_name = "premise"
        data_path = os.path.join("../data", args.dataset, "valid.jsonl")
        label_path = os.path.join("../data", args.dataset, "valid-labels.lst")
        file_path = [data_path, label_path]
        loader = piqa_loader
    elif args.dataset == "qasc":
        ending_names = ['hypothesis0', 'hypothesis1', 'hypothesis2', 'hypothesis3', 'hypothesis4', 'hypothesis5', 'hypothesis6', 'hypothesis7']
        header_name = "premise"
        file_path = os.path.join("../data", args.dataset, "dev.jsonl")
        loader = qasc_loader
    elif args.dataset == "siqa":
        ending_names = ['hypothesis0', 'hypothesis1', 'hypothesis2']
        header_name = "premise"
        data_path = os.path.join("../data", args.dataset, "dev.jsonl")
        label_path = os.path.join("../data", args.dataset, "dev-labels.lst")
        file_path = [data_path, label_path]
        loader = siqa_loader
    elif args.dataset == "winogrande":
        ending_names = ['hypothesis0', 'hypothesis1']
        header_name = "premise"
        data_path = os.path.join("../data", args.dataset, "dev.jsonl")
        label_path = os.path.join("../data", args.dataset, "dev-labels.lst")
        file_path = [data_path, label_path]
        loader = winogrande_loader
    else:
        print(f"{args.dataset}: downloader not implemented.")
        return

    
    dev_data = loader(file_path, args)
    dataset = Dataset.from_list(dev_data).with_format("torch")
    return ending_names, header_name, dataset

def load_model(device, model_path, args):
    if args.model_family in ["GPT2","Pythia"]:
        tokenizer_func = AutoTokenizer
        model_func = AutoModelForCausalLM
    elif args.model_family in ["T5", "FLAN-T5"]:
        tokenizer_func = AutoTokenizer
        model_func = AutoModelForSeq2SeqLM
    else:
        print(f"{args.model_family}: downloader not implemented.")
        return
    tokenizer = tokenizer_func.from_pretrained(model_path)
    if args.model_family in ["GPT2", "Pythia"]:
        tokenizer.pad_token = tokenizer.eos_token
    model = model_func.from_pretrained(model_path)
    model.to(device)
    return model, tokenizer

def write_to_csv(save_path, args, total_accuracy):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    csv_exists = os.path.isfile(save_path)
    with open(save_path, 'a+', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        if not csv_exists:
            csvwriter.writerow(['model_family', 'checkpoint', 'dataset', 'batch_size', 'method', "seed", 'accuracy'])
        csvwriter.writerow([args.model_family, args.checkpoint, args.dataset, args.batch_size, args.method, args.seed, f"{total_accuracy:.4f}"])