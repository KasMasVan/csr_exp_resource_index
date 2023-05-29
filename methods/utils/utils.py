import argparse
import csv
import logging
import os
import random
import sys
from tqdm import tqdm

import numpy as np
import torch
import transformers
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

    conceptual_combinations_loader,
    emoji_movie_loader,
    ruin_names_loader,

    anli_loader
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
        choices=["GPT2", "T5", "FLAN-T5", "Pythia", "OPT-IML", "Dolly"],
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
        "--loading_precision",
        type=str,
        choices=["FP32", "FP16", "BF16", "INT8"],
        default="FP32",
        help="The precision of the model to be loaded."
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
        "--sample",
        type=int,
        default=None,
        help="The number of samples to inference on. If None, inference on the whole dataset.",
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
    parser.add_argument(
        "--calibration_prompt",
        type=str,
        default=None,
        help="Calibration prompt, as in P(y|x)/P(y|calibration_prompt).",
    )
    parser.add_argument(
        "--do_channel",
        action="store_true",
        help="Whether to do channel, i.e., P(x|y_i).",
    )
    parser.add_argument(
        "--process_of_elimination_prompt",
        type=str,
        default=None,
        help="The process of elimination prompt. It asks the model to ignore masked options.",
    )
    parser.add_argument(
        "--scoring_method_for_process_of_elimination",
        type=str,
        choices=["language_modeling", "calibration", "channel", "multiple_choice_prompt"],
        default="language_modeling",
        help="The scoring method for process of elimination.",
    )
    parser.add_argument(
        "--prompting_method_for_process_of_elimination",
        type=str,
        choices=["multiple_choice_prompt"],
        default="multiple_choice_prompt",
        help="The prompting method for process of elimination.",
    )
    parser.add_argument(
        "--mask_strategy_for_process_of_elimination",
        type=str,
        choices=["lowest", "below_average"],
        default="lowest",
        help="The mask strategy for process of elimination.",
    )
    parser.add_argument(
        "--do_synonym",
        action="store_true",
        help="Whether to generate synonyms for options.",
    )
    parser.add_argument(
        "--number_of_synonyms",
        type=int,
        default=5,
        help="The number of synonyms to be used in the generative method.",
    )
    parser.add_argument(
        "--generate_synonyms_prompt",
        type=str,
        default=None,
        help="The prompt template for generating synonyms. 'option is replaced with actual options'",
    )
    parser.add_argument(
        "--push_data_to_hub",
        action="store_true",
        help="Whether to push the data to Hugging Face Hub. This is convienient for LLM experiments.",
    )

    args = parser.parse_args()
    return args

def load_data(args):
    # load test data for final performance.
    # load dev data to tune hyperparameters.
    if args.dataset == "copa":
        ending_names = ['hypothesis0', 'hypothesis1']
        header_name = "premise"
        file_path = os.path.join("../data", args.dataset, "copa-test.xml")
        loader = copa_loader
    elif args.dataset == "cqa":
        ending_names = ['hypothesis0', 'hypothesis1', 'hypothesis2', 'hypothesis3', 'hypothesis4']
        header_name = "premise"
        file_path = os.path.join("../data", args.dataset, "dev.jsonl")
        loader = cqa_loader
    elif args.dataset == "obqa":
        ending_names = ['hypothesis0', 'hypothesis1', 'hypothesis2', 'hypothesis3']
        header_name = "premise"
        file_path = os.path.join("../data", args.dataset, "test.jsonl")
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
    # BIG-Bench tasks
    elif args.dataset == "conceptual_combinations":
        ending_names = ['hypothesis0', 'hypothesis1', 'hypothesis2', 'hypothesis3']
        header_name = "premise"
        file_path = []
        file_suffixes = ["contradictions", "emergent_properties", "fanciful_fictional_combinations", "homonyms", "invented_words", "surprising_uncommon_combinations"]
        for suffix in file_suffixes:
            file_path.append(os.path.join("../data", "big_bench", f"{args.dataset}_{suffix}.json"))
        loader = conceptual_combinations_loader
    elif args.dataset == "emoji_movie":
        ending_names = ['hypothesis0', 'hypothesis1', 'hypothesis2', 'hypothesis3', 'hypothesis4']
        header_name = "premise"
        file_path = os.path.join("../data", "big_bench", f"{args.dataset}.json")
        loader = emoji_movie_loader
    elif args.dataset in ["ruin_names", "temporal_sequences"]:
        ending_names = ['hypothesis0', 'hypothesis1', 'hypothesis2', 'hypothesis3']
        header_name = "premise"
        file_path = os.path.join("../data", "big_bench", f"{args.dataset}.json")
        loader = ruin_names_loader
    elif args.dataset == "strange_stories":
        ending_names = ['hypothesis0', 'hypothesis1', 'hypothesis2', 'hypothesis3']
        header_name = "premise"
        file_path = os.path.join("../data", "big_bench", f"{args.dataset}_multiple_choice.json")
        loader = ruin_names_loader
    elif args.dataset == "anli":
        ending_names = ['hypothesis0', 'hypothesis1', 'hypothesis2']
        header_name = "premise"
        file_path = []
        file_prefixes = ["R1", "R2", "R3"]
        for prefix in file_prefixes:
            file_path.append(os.path.join("../data", f"{args.dataset}", f"{prefix}_dev.jsonl"))
        loader = anli_loader
    else:
        print(f"{args.dataset}: downloader not implemented.")
        return

    
    dev_data = loader(file_path, args)
    dataset = Dataset.from_list(dev_data).with_format("torch")
    return ending_names, header_name, dataset

def load_model(device, model_path, args):
    if args.model_family in ["GPT2","Pythia", "OPT-IML", "Dolly"]:
        tokenizer_func = AutoTokenizer
        model_func = AutoModelForCausalLM
    elif args.model_family in ["T5", "FLAN-T5"]:
        tokenizer_func = AutoTokenizer
        model_func = AutoModelForSeq2SeqLM
    else:
        print(f"{args.model_family}: downloader not implemented.")
        return
    if args.model_family == "Dolly":
        tokenizer = tokenizer_func.from_pretrained(model_path, padding_side="left")
    else:
        tokenizer = tokenizer_func.from_pretrained(model_path)
    if args.model_family in ["GPT2", "Pythia", "Dolly"]:
        tokenizer.pad_token = tokenizer.eos_token
    # load with different precision
    if args.loading_precision == "FP16":
        model = model_func.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)
    elif args.loading_precision == "BF16":
        model = model_func.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16)
    elif args.loading_precision == "INT8":
        model = model_func.from_pretrained(model_path, device_map="auto", load_in_8bit=True)
    else: # FP32
        model = model_func.from_pretrained(model_path)
        model.to(device)
    print(f"Memory footprint: {model.get_memory_footprint() / 1024 **3:.2f} GB.")
    return model, tokenizer

def write_to_csv(save_path, args, total_accuracy):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    csv_exists = os.path.isfile(save_path)
    with open(save_path, 'a+', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        if args.method == "process_of_elimination":
            if not csv_exists:
                csvwriter.writerow(['model_family', 'checkpoint', 'loading_precision','dataset', 'batch_size', 'method', "scoring_method", "prompting_method", "mask_strategy", "seed", "sample", "mask_accuracy", 'accuracy'])
            csvwriter.writerow([args.model_family, args.checkpoint, args.loading_precision, args.dataset, args.batch_size, args.method, args.scoring_method_for_process_of_elimination, args.prompting_method_for_process_of_elimination, args.mask_strategy_for_process_of_elimination, args.seed, args.sample, f"{args.mask_accuracy:.4f}", f"{total_accuracy:.4f}"])
        elif args.method == "generate_synonyms":
            if not csv_exists:
                csvwriter.writerow(['model_family', 'checkpoint', 'loading_precision','dataset', 'batch_size', 'method', "number_of_synonyms", "seed", "sample",'accuracy'])
            csvwriter.writerow([args.model_family, args.checkpoint, args.loading_precision, args.dataset, args.batch_size, args.method, args.number_of_synonyms, args.seed, args.sample, f"{total_accuracy:.4f}"])
        else:
            if not csv_exists:
                csvwriter.writerow(['model_family', 'checkpoint', 'loading_precision','dataset', 'batch_size', 'method', "seed", "sample",'accuracy'])
            csvwriter.writerow([args.model_family, args.checkpoint, args.loading_precision, args.dataset, args.batch_size, args.method, args.seed, args.sample, f"{total_accuracy:.4f}"])