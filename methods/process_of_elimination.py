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

def compute_mask(model, eval_dataloader, device, compute_func):
    model.eval()
    masks = []
    torch.cuda.empty_cache()

    pbar = tqdm(eval_dataloader, desc="Computing masks")
    for batch in pbar:
        # -logP(ending|header)
        log_prob = compute_func(batch, model, device)
        avg_log_prob = log_prob / batch["ending_attention_mask"].sum(dim=-1)
        
        # soft masking, i.e., get rid of the least likely answer.
        # mask = torch.ones_like(log_prob)
        # mask[torch.arange(avg_log_prob.shape[0]), avg_log_prob.argmax(dim=-1)] = 0
        # masks.append(mask)

        # soft masking v2, i.e., get rid of the answers that are below the mean.
        mask_v2 = torch.ones_like(log_prob)
        # Calculate the row-wise mean
        row_mean = avg_log_prob.mean(dim=1, keepdim=True)
        # Set values below the mean to 0
        mask_v2[avg_log_prob > row_mean] = 0
        masks.append(mask_v2)

    masks = torch.cat(masks, dim=0)
    return masks

def inference_process_of_elimination(model, eval_dataloader, device, compute_func):
    model.eval()
    lm_predictions = torch.zeros(0)
    avg_lm_predictions = torch.zeros(0)
    labels = torch.zeros(0)
    torch.cuda.empty_cache()

    pbar = tqdm(eval_dataloader, desc="Inference")
    for batch in pbar:
        log_prob = compute_func(batch, model, device)
        # apply hard masking
        log_prob[batch["mask"] == 0] = float("inf")
        
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

def create_multiple_choice_prompt(example, **kwargs):
    alphabets = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    multiple_choice_prompt = kwargs["multiple_choice_prompt"]
    mask = example['mask']
    mcp_example = {}
    # example['premise'] = premise = f"{multiple_choice_prompt} {premise}\nA. {options[0]}\nB. {options[1]}\nC. {options[2]}\nD. {options[3]}\nE. {options[4]}\nAnswer:"
    premise = f"{multiple_choice_prompt} {example['premise']}\n"
    for idx, single_mask in enumerate(mask):
        mcp_example[f'hypothesis{idx}'] = alphabets[idx]
        if single_mask == 1:
            premise += f"{alphabets[idx]}. {example[f'hypothesis{idx}']}\n"
        else:
            # consider other null strings.
            premise += f"{alphabets[idx]}. [MASK]\n"
    premise += "Answer:"
    mcp_example['premise'] = premise
    return mcp_example
    

def main():
    # import pdb; pdb.set_trace()

    # step 1: argument parser, and logger
    args = parse_args()
    args.method = "process_of_elimination"

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
    if args.model_family in ["GPT2", "Pythia"]:
        compute_func = compute_conditional_score_causal
        preprocess_func = preprocess_function_causal
        remove_columns = ['input_ids',
                          'labels',
                          'ending_attention_mask']
    elif args.model_family in ["T5", "FLAN-T5"]:
        compute_func = compute_conditional_score_seq2seq
        preprocess_func = preprocess_function_seq2seq
        remove_columns=['header_input_ids', 
                        'header_attention_mask', 
                        'ending_input_ids', 
                        'ending_attention_mask', ]
    else:
        raise NotImplementedError

    # step 4: load and preprocess data.
    args.datasets = args.datasets.split()
    logger.info(f"Load data: {args.datasets}.")
    
    # evaluate on each dataset
    for dataset in args.datasets:
        args.dataset = dataset
        multiple_choice_prompt = args.multiple_choice_prompt
        args.multiple_choice_prompt = None
        ending_names, header_name, raw_dataset = load_data(args)

        logger.info(f"Preprocess data: {args.dataset}.")
        fn_kwargs = {"ending_names": ending_names, 
                    "header_name": header_name, 
                    "tokenizer": tokenizer,}
        tokenized_dataset = raw_dataset.map(preprocess_func, fn_kwargs=fn_kwargs, batched=True, batch_size=args.batch_size)
        eval_dataloader = DataLoader(tokenized_dataset, batch_size=args.batch_size, shuffle=False)

        # step 5: (evaluation) inference on data, and compute accuracy.
        logger.info(f"Start inference (method: {args.method}) on {args.dataset} using {args.model_family} model: {args.checkpoint}.")
        logger.info(f"Step 1: Computing masks.")
        masks = compute_mask(model, eval_dataloader, device, compute_func)
        masked_dataset = tokenized_dataset.map(lambda example, idx: {"mask": masks[idx]}, 
                                 with_indices=True, 
                                 batched=True,
                                 remove_columns=remove_columns)
        
        logger.info(f"Step 2: Creating multiple choice prompt.")
        mcp_kwargs = {"multiple_choice_prompt": multiple_choice_prompt,}
        mcp_dataset = masked_dataset.map(create_multiple_choice_prompt, fn_kwargs=mcp_kwargs)
        
        logger.info(f"Step 3: Final Inference")
        mcp_dataset = mcp_dataset.map(preprocess_func, fn_kwargs=fn_kwargs, batched=True, batch_size=args.batch_size)
        eval_mcp_dataloader = DataLoader(mcp_dataset, batch_size=args.batch_size, shuffle=False)
        lm_accuracy, _ = inference_process_of_elimination(model, eval_mcp_dataloader, device, compute_func)

        # step 6: some postprocessing, including saving and displyaing output.
        save_path = os.path.join("../results", f"{args.method}.csv")
        logger.info(f"Save results to {save_path}.")
        write_to_csv(save_path, args, lm_accuracy)

if __name__ == "__main__":
    main()