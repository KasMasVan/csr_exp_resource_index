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
from datasets import Dataset

from utils.data import(
    upload_to_huggingface_hub,
    preprocess_function_seq2seq,
    preprocess_function_causal,
    preprocess_function_causal_channel,
    preprocess_function_seq2seq_channel,
    create_synonym_dataset,
)
from utils.methods import(
    compute_conditional_score_seq2seq,
    compute_conditional_score_causal,
    inference_language_modeling,
    inference_calibration,
    inference_generate_synonyms,
    generate_synonyms,
)
from utils.utils import(
    load_data,
    load_model,
    parse_args,
    set_seed,
    write_to_csv,
)

logger = logging.getLogger(__name__)

def main():
    import pdb; pdb.set_trace()

    # step 1: argument parser, and logger
    args = parse_args()
    # if args.multiple_choice_prompt is not None:
    #     args.method = "multiple_choice_prompt"
    # elif args.calibration_prompt is not None:
    #     args.method = "calibration"
    # elif args.do_channel == True:
    #     args.method = "channel"
    # else:
    #     args.method = "language_modeling"
    args.method = "generate_synonyms"

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
    if args.model_family in ["GPT2", "Pythia", "OPT-IML", "Dolly"]:
        compute_func = compute_conditional_score_causal
        preprocess_func = preprocess_function_causal
        preprocess_func_channel = preprocess_function_causal_channel
    elif args.model_family in ["T5", "FLAN-T5"]:
        compute_func = compute_conditional_score_seq2seq
        preprocess_func = preprocess_function_seq2seq
        preprocess_func_channel = preprocess_function_seq2seq_channel
    else:
        raise NotImplementedError

    # step 4: load and preprocess data.
    args.datasets = args.datasets.split()
    logger.info(f"Load data: {args.datasets}.")
    
    # evaluate on each dataset
    for dataset in args.datasets:
        args.dataset = dataset
        ending_names, header_name, raw_dataset = load_data(args)
        if args.sample is not None:
            # sample "sample" amount of data from raw_data
            raw_dataset = raw_dataset.shuffle(seed=args.seed).select(range(args.sample))

        logger.info(f"Preprocess data: {args.dataset}.")
        fn_kwargs = {"ending_names": ending_names, 
                    "header_name": header_name, 
                    "tokenizer": tokenizer,}
        num_of_options = len(ending_names)
        tokenized_dataset = raw_dataset.map(preprocess_func, fn_kwargs=fn_kwargs, batched=True, batch_size=args.batch_size)
        eval_dataloader = DataLoader(tokenized_dataset, batch_size=args.batch_size, shuffle=False)

        # step 5: (evaluation) inference on data, and compute accuracy.
        logger.info(f"Start inference (method: {args.method}) on {args.dataset} using {args.model_family} model: {args.checkpoint}.")
        if args.method in ["language_modeling", "multiple_choice_prompt"]:
            _, lm_accuracy, avg_lm_accuracy = inference_language_modeling(model, eval_dataloader, device, compute_func, tokenizer.pad_token_id)
        elif args.method == "calibration":
            fn_kwargs = {"ending_names": ending_names, 
                        "header_name": "uncond_premise", # the difference is here
                        "tokenizer": tokenizer,}
            tokenized_calibration_dataset = raw_dataset.map(preprocess_func, fn_kwargs=fn_kwargs, batched=True, batch_size=args.batch_size)
            eval_calibration_dataloader = DataLoader(tokenized_calibration_dataset, batch_size=args.batch_size, shuffle=False)    
            _, lm_accuracy, avg_lm_accuracy = inference_calibration(model, eval_dataloader, eval_calibration_dataloader,device, compute_func, tokenizer.pad_token_id)
        elif args.method == "channel":
            # simple solution: swap first sentence and second sentence in both preprocessing functions
            tokenized_channel_dataset = raw_dataset.map(preprocess_func_channel, fn_kwargs=fn_kwargs, batched=True, batch_size=args.batch_size)
            eval_channel_dataloader = DataLoader(tokenized_channel_dataset, batch_size=args.batch_size, shuffle=False)
            _, lm_accuracy, avg_lm_accuracy = inference_language_modeling(model, eval_channel_dataloader, device, compute_func, tokenizer.pad_token_id)
        elif args.method == "generate_synonyms":
            # 3 stpes: generate synonyms, then map datasets, then inference.
            logger.info(f"Generate synonyms for {args.dataset}.")
            synonyms_dict = generate_synonyms(args, model, tokenizer, tokenized_dataset)
            # call map add synonyms to raw_dataset
            logger.info(f"Add synonyms to raw dataset.")
            synonym_kwargs = {"args": args,
                              "synonyms_dict": synonyms_dict,
                              }
            synonyms_dataset = raw_dataset.map(create_synonym_dataset, fn_kwargs=synonym_kwargs, batched=True, batch_size=args.batch_size)
            # map to tokenized_dataset
            logger.info(f"Tokenize synonym data.")
            synonyms_ending_names = [col for col in synonyms_dataset.column_names if col.startswith("hypothesis")]
            fn_kwargs = {"ending_names": synonyms_ending_names, 
                        "header_name": header_name, 
                        "tokenizer": tokenizer,}
            tokenized_synonyms_dataset = synonyms_dataset.map(preprocess_func, fn_kwargs=fn_kwargs, batched=True, batch_size=args.batch_size)
            eval_synonyms_dataloader = DataLoader(tokenized_synonyms_dataset, batch_size=args.batch_size, shuffle=False)
            _, lm_accuracy, avg_lm_accuracy = inference_generate_synonyms(model, eval_synonyms_dataloader, device, compute_func, tokenizer.pad_token_id, num_of_options, args.number_of_synonyms)
        else:
            raise NotImplementedError

        # step 6: some postprocessing, including saving and displyaing output.
        save_path = os.path.join("../results", f"{args.method}.csv")
        logger.info(f"Save results to {save_path}.")
        write_to_csv(save_path, args, lm_accuracy)

        # does calibration needs copying as well?
        if args.method == "language_modeling":
            avg_args = copy.deepcopy(args)
            avg_args.method = "average_language_modeling"
            write_to_csv(save_path, avg_args, avg_lm_accuracy)
        
        # step 7: push data to HuggingFace Hub.
        if args.push_data_to_hub:
            logger.info(f"Push {args.dataset} to HuggingFace Hub.")
            upload_to_huggingface_hub(tokenized_dataset, args)
        
        # step 8: delete tokenized_dataset to save memory.
        # del tokenized_dataset
            

if __name__ == "__main__":
    main()