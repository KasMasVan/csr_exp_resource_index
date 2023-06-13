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
    create_n_shot_splits,
)
from utils.methods import(
    compute_conditional_score_seq2seq,
    compute_conditional_score_causal,
    inference_language_modeling,
    inference_calibration,
    inference_contrastive_decoding,
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
    # import pdb; pdb.set_trace()

    # step 1: argument parser, and logger
    args = parse_args()
    args.method = "contrastive_decoding"

    # handle different methods, this is a provisional solution
    for method in [args.expert_method, args.amateur_method]:
        if method == "multiple_choice_prompt":
            args.multiple_choice_prompt = ""
        elif method == "calibration":
            args.calibration_prompt = " the answer is:"
        elif method == "channel":
            pass
        elif method == "language_modeling":
            pass

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
    logger.info(f"Load {args.model_family} expert model: {args.checkpoint}.")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # get model path: ../models/args.model_family/args.checkpoint
    model_path = os.path.join("../models", args.model_family, args.checkpoint)
    model, tokenizer = load_model(device, model_path, args)
    logger.info(f"Load {args.model_family} amateur model: {args.amateur_checkpoint}.")
    amateur_model_path = os.path.join("../models", args.model_family, args.amateur_checkpoint)
    amateur_model, _ = load_model(device, amateur_model_path, args)
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
        ending_names, header_name, raw_dataset, n_shot_dataset = load_data(args)
        raw_dataset, n_shot_dataset, n_shot_demonstrations = create_n_shot_splits(raw_dataset, n_shot_dataset, args)    

        logger.info(f"Preprocess data: {args.dataset}.")
        
        # CD special treatment: the expert model and amateur model may use different methods
        # step 5: (evaluation) inference on data, and compute accuracy.
        logger.info(f"Start inference (expert method: {args.expert_method}")
        method_agnostic_kwargs = {
            "args": args,
            "raw_dataset": raw_dataset,
            "device": device,
            "compute_func": compute_func,
            "tokenizer": tokenizer,
            "ending_names": ending_names,
            "header_name": header_name,
            "preprocess_func": preprocess_func,
            "preprocess_func_channel": preprocess_func_channel,
        }
        exp_avg_log_probs, exp_lm_accuracy, exp_avg_lm_accuracy = inference_contrastive_decoding(args.expert_method, model, **method_agnostic_kwargs) 
        logger.info(f"Start inference (amateur method: {args.amateur_method}")
        ama_avg_log_probs, ama_lm_accuracy, ama_avg_lm_accuracy = inference_contrastive_decoding(args.amateur_method, amateur_model, **method_agnostic_kwargs)  
        # weighting_parameter = args.weighting_parameter
        if args.num_random_search == 0:
            weighting_parameters = [args.weighting_parameter] # -1
        else:
            # sample from [-2, 0]
            weighting_parameters = np.random.uniform(-2, 0, args.num_random_search)

        for weighting_parameter in weighting_parameters:
            args.weighting_parameter = weighting_parameter
            logger.info(f"weighting parameter: {weighting_parameter}")
            avg_log_probs = exp_avg_log_probs + weighting_parameter * ama_avg_log_probs
            labels = raw_dataset['label']
            lm_accuracy = (avg_log_probs.argmin(dim=-1) == labels).sum().item() / len(labels)
            logger.info(f"Contrastive decoding accuracy: {lm_accuracy:.4f}.")
            args.amateur_accuracy = ama_avg_lm_accuracy
            args.expert_accuracy = exp_avg_lm_accuracy
            # logger.info(f"Start inference (method: {args.method}) on {args.dataset} using {args.model_family} model: {args.checkpoint}.")

            # step 6: some postprocessing, including saving and displyaing output.
            save_path = os.path.join("../results", f"{args.method}.csv")
            logger.info(f"Save results to {save_path}.")
            write_to_csv(save_path, args, lm_accuracy)

            # does calibration needs copying as well?
            # if args.method == "language_modeling":
            #     avg_args = copy.deepcopy(args)
            #     avg_args.method = "average_language_modeling"
            #     write_to_csv(save_path, avg_args, avg_lm_accuracy)
            
            # step 7: push data to HuggingFace Hub.
            # if args.push_data_to_hub:
            #     logger.info(f"Push {args.dataset} to HuggingFace Hub.")
            #     upload_to_huggingface_hub(tokenized_dataset, args)
            
            # step 8: delete tokenized_dataset to save memory.
            # del tokenized_dataset
            

if __name__ == "__main__":
    main()