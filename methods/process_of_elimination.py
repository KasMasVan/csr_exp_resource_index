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
    upload_to_huggingface_hub,
    preprocess_function_seq2seq,
    preprocess_function_causal,
)
from utils.methods import(
    compute_conditional_score_seq2seq,
    compute_conditional_score_causal,
    compute_mask_process_of_elimination,
    inference_process_of_elimination,
)
from utils.utils import(
    load_data,
    load_model,
    parse_args,
    set_seed,
    write_to_csv,
)

logger = logging.getLogger(__name__)

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
    if args.model_family in ["GPT2", "Pythia", "OPT-IML"]:
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
        if args.sample is not None:
            # sample "sample" amount of data from raw_data
            raw_dataset = raw_dataset.shuffle(seed=args.seed).select(range(args.sample))

        logger.info(f"Preprocess data: {args.dataset}.")
        fn_kwargs = {"ending_names": ending_names, 
                    "header_name": header_name, 
                    "tokenizer": tokenizer,}
        tokenized_dataset = raw_dataset.map(preprocess_func, fn_kwargs=fn_kwargs, batched=True, batch_size=args.batch_size)
        eval_dataloader = DataLoader(tokenized_dataset, batch_size=args.batch_size, shuffle=False)

        # step 5: (evaluation) inference on data, and compute accuracy.
        logger.info(f"Start inference (method: {args.method}) on {args.dataset} using {args.model_family} model: {args.checkpoint}.")
        logger.info(f"Step 1: Computing masks.")
        # if args.scoring_method_for_process_of_elimination
        masks = compute_mask_process_of_elimination(model, eval_dataloader, device, compute_func, tokenizer.pad_token_id)
        masks = masks.to(torch.float32)
        masked_dataset = tokenized_dataset.map(lambda example, idx: {"mask": masks[idx]}, 
                                 with_indices=True, 
                                 batched=True,
                                 remove_columns=remove_columns)
        
        logger.info(f"Step 2: Creating multiple choice prompt.")
        # if args.prompting_method_for_process_of_elimination
        mcp_kwargs = {"multiple_choice_prompt": multiple_choice_prompt,}
        mcp_dataset = masked_dataset.map(create_multiple_choice_prompt, fn_kwargs=mcp_kwargs)
        
        logger.info(f"Step 3: Final Inference")
        mcp_dataset = mcp_dataset.map(preprocess_func, fn_kwargs=fn_kwargs, batched=True, batch_size=args.batch_size)
        eval_mcp_dataloader = DataLoader(mcp_dataset, batch_size=args.batch_size, shuffle=False)
        lm_accuracy, _ = inference_process_of_elimination(model, eval_mcp_dataloader, device, compute_func, tokenizer.pad_token_id)

        # step 6: some postprocessing, including saving and displyaing output.
        save_path = os.path.join("../results", f"{args.method}.csv")
        logger.info(f"Save results to {save_path}.")
        write_to_csv(save_path, args, lm_accuracy)

        # step 7: push data to HuggingFace Hub.
        if args.push_data_to_hub:
            logger.info(f"Push {args.dataset} to HuggingFace Hub.")
            # save the mcp dataset, which will be used by LLM.
            upload_to_huggingface_hub(mcp_dataset, args)

if __name__ == "__main__":
    main()