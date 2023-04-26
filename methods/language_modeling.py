# a framework for inference on multiple choice tasks.
import argparse
import copy
import csv
import logging
import os
import random
import sys
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import(
    AutoTokenizer, 
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)
from datasets import Dataset


from utils.data import(
    preprocess_function_seq2seq,
    preprocess_function_causal,
)
from utils.methods import(
    compute_conditional_score_seq2seq,
    compute_conditional_score_causal,
)
from utils.utils import(
    load_data,
    load_model,
    parse_args,
    set_seed,
    write_to_csv,
)

logger = logging.getLogger(__name__)

def inference_language_modeling(model, eval_dataloader, device, compute_func):
    model.eval()
    lm_predictions = torch.zeros(0)
    avg_lm_predictions = torch.zeros(0)
    labels = torch.zeros(0)
    torch.cuda.empty_cache()

    pbar = tqdm(eval_dataloader, desc="Inference")
    for batch in pbar:
        log_prob = compute_func(batch, model, device)
        
        ending_length = batch["ending_attention_mask"].sum(dim=-1)
        batch_predictions = log_prob.argmin(dim=-1)
        batch_avg_predictions = (log_prob / ending_length).argmin(dim=-1)

        batch_labels = batch["label"]
        lm_predictions = torch.cat((lm_predictions, batch_predictions))
        avg_lm_predictions = torch.cat((avg_lm_predictions, batch_avg_predictions))
        labels = torch.cat((labels, batch_labels))
    
        # make accuracy accumulative
        lm_accuracy = (lm_predictions == labels).sum().item() / len(labels)
        avg_lm_accuracy = (avg_lm_predictions == labels).sum().item() / len(labels)
        pbar.set_description(f"Language modeling accuracy: {lm_accuracy:.4f}, Average language modeling accuracy: {avg_lm_accuracy:.4f}")
    return lm_accuracy, avg_lm_accuracy

def main():
    # import pdb; pdb.set_trace()

    # step 1: argument parser, and logger
    args = parse_args()
    if args.multiple_choice_prompt is not None:
        args.method = "multiple_choice_prompt"
    else:
        args.method = "language_modeling"

    # print(args)
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # get model path: ../models/args.model_family/args.checkpoint
    model_path = os.path.join("../models", args.model_family, args.checkpoint)
    model, tokenizer = load_model(device, model_path, args)
    if args.model_family in ["GPT2"]:
        compute_func = compute_conditional_score_causal
        preprocess_func = preprocess_function_causal

    elif args.model_family in ["T5", "FLAN-T5"]:
        compute_func = compute_conditional_score_seq2seq
        preprocess_func = preprocess_function_seq2seq
    else:
        raise NotImplementedError

    # step 4: load and preprocess data.
    args.datasets = args.datasets.split()
    logger.info(f"Load data: {args.datasets}.")
    
    # evaluate on each dataset
    for dataset in args.datasets:
        args.dataset = dataset
        ending_names, header_name, raw_dataset = load_data(args)

        logger.info(f"Preprocess data: {args.dataset}.")
        fn_kwargs = {"ending_names": ending_names, 
                    "header_name": header_name, 
                    "tokenizer": tokenizer,}
        tokenized_dataset = raw_dataset.map(preprocess_func, fn_kwargs=fn_kwargs, batched=True, batch_size=args.batch_size)
        eval_dataloader = DataLoader(tokenized_dataset, batch_size=args.batch_size, shuffle=False)

        # step 5: (evaluation) inference on data, and compute accuracy.
        logger.info(f"Start inference (method: {args.method}) on {args.dataset} using {args.model_family} model: {args.checkpoint}.")
        lm_accuracy, avg_lm_accuracy = inference_language_modeling(model, eval_dataloader, device, compute_func)
    
        # step 6: some postprocessing, including saving and displyaing output.
        save_path = os.path.join("../results", f"{args.method}.csv")
        logger.info(f"Save results to {save_path}.")
        write_to_csv(save_path, args, lm_accuracy)

        if args.method == "language_modeling":
            avg_args = copy.deepcopy(args)
            avg_args.method = "average_language_modeling"
            write_to_csv(save_path, avg_args, avg_lm_accuracy)

if __name__ == "__main__":
    main()