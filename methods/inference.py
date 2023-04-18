# a framework for inference on multiple choice tasks.
import argparse
import csv
import logging
import os
import random
import sys
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from transformers import(
    AutoTokenizer, 
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    DataCollatorWithPadding
)
from datasets import Dataset


from utils.data import(
    copa_loader,
    cqa_loader,
    winogrande_loader,
)
from utils.methods import(
    inference_language_modeling,
    inference_contrastive_decoding,
)
from utils.models import(
    find_expert_model,
)

logger = logging.getLogger(__name__)

def preprocess_function(examples, **kwargs):
    ending_names, header_name, tokenizer = kwargs['ending_names'], kwargs['header_name'], kwargs['tokenizer']
    num_choice = len(ending_names)
    question_headers = examples[header_name]
    # the tokenizer handles multiple spaces.
    first_sentences = [[context] * len(ending_names) for context in examples[header_name]]
    # second_sentences = [
    #     [f"{header} {examples[end][i]}" for end in ending_names] for i, header in enumerate(question_header)
    # ]
    second_sentences = [
        [f"{examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)
    ]

    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])

    # tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)
    tokenized_headers = tokenizer(first_sentences, padding=True, truncation=True)
    tokenized_endings = tokenizer(second_sentences, padding=True, truncation=True)
    header_dict = {f"header_{k}": [v[i : i + num_choice] for i in range(0, len(v), num_choice)] for k, v in tokenized_headers.items()}
    ending_dict = {f"ending_{k}": [v[i : i + num_choice] for i in range(0, len(v), num_choice)] for k, v in tokenized_endings.items()}
    return {**header_dict, **ending_dict}

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
        choices=["GPT2", "T5", "FLAN-T5"],
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
        "--data",
        type=str,
        choices=["copa", "cqa", "winogrande"],
        default=None,
        required=True,
        help="The dataset to inference on.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["all", "language_modeling", "contrastive_decoding"],
        default=None,
        required=True,
        help="The inference method. Choose all to use all avaiable methods."
    )

    args = parser.parse_args()
    return args

def write_to_csv(save_path, args, total_accuracy):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    csv_exists = os.path.isfile(save_path)
    with open(save_path, 'a+', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        if not csv_exists:
            csvwriter.writerow(['model_family', 'checkpoint', 'data', 'batch_size', 'method', "seed", 'accuracy'])
        csvwriter.writerow([args.model_family, args.checkpoint, args.data, args.batch_size, args.method, args.seed, f"{total_accuracy:.4f}"])

def write_to_csv_contrastive_decoding(save_path, args, expert_checkpoint, total_accuracy):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    csv_exists = os.path.isfile(save_path)
    with open(save_path, 'a+', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        if not csv_exists:
            csvwriter.writerow(['model_family', 'amateur_checkpoint', 'expert_checkpoint', 'data', 'batch_size', 'method', "seed", 'accuracy'])
        csvwriter.writerow([args.model_family, args.checkpoint, expert_checkpoint, args.data, args.batch_size, args.method, args.seed, f"{total_accuracy:.4f}"])


def main():
    import pdb; pdb.set_trace()

    # step 1: argument parser, and logger
    args = parse_args()
    print(args)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.setLevel(logging.INFO)

    # step 2: set random seed to ensure reproducibility.
    logger.info(f"Set random seed to {args.seed}.")
    set_seed(args.seed)

    # step 3: load model, tokenizer. Then move to gpu, and set to evaluation mode.
    logger.info(f"Load {args.model_family} model: {args.checkpoint}.")
    # torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # get model path: ../models/args.model_family/args.checkpoint
    model_path = os.path.join("../models", args.model_family, args.checkpoint)
    if args.model_family == "GPT2":
        tokenizer_func = AutoTokenizer
        model_func = AutoModelForCausalLM
    elif args.model_family in ["T5", "FLAN-T5"]:
        tokenizer_func = AutoTokenizer
        model_func = AutoModelForSeq2SeqLM
    else:
        print(f"{args.model_family}: downloader not implemented.")
        return
    tokenizer = tokenizer_func.from_pretrained(model_path)
    model = model_func.from_pretrained(model_path)
    model.to(device)

    # step 4: load and preprocess data.
    logger.info(f"Load data: {args.data}.")
    if args.data == "copa":
        ending_names = ['hypothesis0', 'hypothesis1']
        header_name = "premise"
        file_path = os.path.join("../data", args.data, "copa-dev.xml")
        loader = copa_loader
    elif args.data == "cqa":
        ending_names = ['hypothesis0', 'hypothesis1', 'hypothesis2', 'hypothesis3', 'hypothesis4']
        header_name = "premise"
        file_path = os.path.join("../data", args.data, "dev.jsonl")
        loader = cqa_loader
    elif args.data == "winogrande":
        file_path = os.path.join("../data", args.data, "dev.jsonl")
        loader = winogrande_loader
    
    dev_data = loader(file_path)
    dataset = Dataset.from_list(dev_data).with_format("torch")
    

    logger.info(f"Preprocess data: {args.data}.")
    fn_kwargs = {"ending_names": ending_names, 
                 "header_name": header_name, 
                 "tokenizer": tokenizer}
    tokenized_dataset = dataset.map(preprocess_function, fn_kwargs=fn_kwargs, batched=True, batch_size=args.batch_size)
    
    eval_dataloader = DataLoader(tokenized_dataset, batch_size=args.batch_size, shuffle=False)

    # step 5: (evaluation) inference on data, and compute accuracy.
    logger.info(f"Start inference (method: {args.method}) on {args.data} using {args.model_family} model: {args.checkpoint}.")
    if args.method == "all":
        logger.info(f"Method {args.method} not implemented yet.")
    elif args.method == "language_modeling":
        total_accuracy = inference_language_modeling(model, eval_dataloader, device)
    elif args.method == "contrastive_decoding":
        # need to instantiate an expert model, e.g., the largest model in the model family.
        amateur_model = model
        # for now, expert checkpoint is hard coded.
        expert_checkpoint = find_expert_model(args.model_family)
        logger.info(f"Load {args.model_family} expert model: {expert_checkpoint}.")
        expert_model_path = os.path.join("../models", args.model_family, expert_checkpoint)
        expert_model = model_func.from_pretrained(expert_model_path)
        expert_model.to(device)
        total_accuracy = inference_contrastive_decoding(amateur_model, expert_model, eval_dataloader, device)
    else:
        # not implemented yet
        logger.info(f"Method {args.method} not implemented yet.")

 
    # step 6: some postprocessing, including saving and displyaing output.

    save_path = os.path.join("../results", f"{args.method}.csv")
    logger.info(f"Save results to {save_path}.")
    if args.method == "contrastive_decoding":
        write_to_csv_contrastive_decoding(save_path, args, expert_checkpoint, total_accuracy)
    else:
        write_to_csv(save_path, args, total_accuracy)

if __name__ == "__main__":
    main()