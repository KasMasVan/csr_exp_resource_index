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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import(
    AutoTokenizer, 
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)
from datasets import Dataset


from utils.data import(
    preprocess_function,
)
# the two methods are commented out, because contrastive decoding
# requires special arguments.
from utils.utils import(
    load_data,
    load_model,
    # parse_args, 
    set_seed,
    # write_to_csv,
)

logger = logging.getLogger(__name__)

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
        "--amateur_checkpoint",
        type=str,
        default=None,
        required=True,
        help="The amateur checkpoint, which is usually a small model.",
    )
    parser.add_argument(
        "--expert_checkpoint",
        type=str,
        default=None,
        required=True,
        help="The expert checkpoint, which is usually a large model from the same model family.",
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

    args = parser.parse_args()
    return args

def inference_contrastive_decoding(amateur_model, expert_model, eval_dataloader, device):
    amateur_model.eval()
    expert_model.eval()
    predictions = torch.zeros(0)
    labels = torch.zeros(0)
    torch.cuda.empty_cache()

    pbar = tqdm(eval_dataloader, desc="Inference")
    for batch in pbar:
        # e.g., (batch_size, #option, ending_seq_len): (32, 2, 18)
        ending_shape = batch["ending_input_ids"].shape 
        # flatten
        header_input_ids = batch["header_input_ids"].view(-1, batch["header_input_ids"].shape[-1]).to(device)
        ending_input_ids = batch["ending_input_ids"].view(-1, batch["ending_input_ids"].shape[-1]).to(device)
        
        # key step: compute logits.
        with torch.no_grad():
            amateur_model_logits = amateur_model(input_ids = header_input_ids, labels = ending_input_ids).logits
            expert_model_logits = expert_model(input_ids = header_input_ids, labels = ending_input_ids).logits
        
        logits = expert_model_logits - amateur_model_logits
        # e.g., (batch_size * #option, ending_seq_len, #vocab): (64, 18, 32128)
        logits = logits.view(-1, logits.shape[-1])
        # ignore padding token: 0
        ce_loss = F.cross_entropy(logits, ending_input_ids.view(-1), reduction="none", ignore_index=0).detach().cpu()
        # each score is the negative log-likelihood of a ending given a header.
        batch_predictions = ce_loss.view(ending_shape).sum(dim=-1).argmin(dim=-1)
        batch_labels = batch["label"]
        predictions = torch.cat((predictions, batch_predictions))
        labels = torch.cat((labels, batch_labels))
    
        # make accuracy accumulative
        batch_accuracy = (batch_predictions == batch_labels).sum().item() / len(batch_labels)
        total_accuracy = (predictions == labels).sum().item() / len(labels)
        pbar.set_description(f"Total Accuracy: {total_accuracy:.4f}, Batch Accuracy: {batch_accuracy:.4f}")
    return total_accuracy

def write_to_csv(save_path, args, total_accuracy):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    csv_exists = os.path.isfile(save_path)
    with open(save_path, 'a+', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        if not csv_exists:
            csvwriter.writerow(['model_family', 'amateur_checkpoint', 'expert_checkpoint', 'dataset', 'batch_size', 'method', "seed", 'accuracy'])
        csvwriter.writerow([args.model_family, args.amateur_checkpoint, args.expert_checkpoint, args.dataset, args.batch_size, args.method, args.seed, f"{total_accuracy:.4f}"])

def main():
    # import pdb; pdb.set_trace()

    # step 1: argument parser, and logger
    args = parse_args()
    args.method = "contrastive_decoding"
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

    # step 3: load two models, tokenizer. Then move to gpu, and set to evaluation mode.
    # the two models should come from the same model family, 
    # so no need to load to two tokenizers.
    logger.info(f"Load {args.model_family} amateur model: {args.amateur_checkpoint} and expert model: {args.expert_checkpoint}.")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # get model path: ../models/args.model_family/args.checkpoint
    amateur_model_path = os.path.join("../models", args.model_family, args.amateur_checkpoint)
    amateur_model, tokenizer = load_model(device, amateur_model_path, args)
    expert_model_path = os.path.join("../models", args.model_family, args.expert_checkpoint)
    expert_model, _ = load_model(device, expert_model_path, args)

    # step 4: load and preprocess data.
    # ending_names, header_name, dataset = load_data(args)
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
        tokenized_dataset = raw_dataset.map(preprocess_function, fn_kwargs=fn_kwargs, batched=True, batch_size=args.batch_size)
        eval_dataloader = DataLoader(tokenized_dataset, batch_size=args.batch_size, shuffle=False)

        # step 5: (evaluation) inference on data, and compute accuracy.
        logger.info(f"Start inference (method: {args.method}) on {args.dataset} using {args.model_family} amateur model: {args.amateur_checkpoint} and expert model: {args.expert_checkpoint}.")
        total_accuracy = inference_contrastive_decoding(amateur_model, expert_model, eval_dataloader, device)

        # step 6: some postprocessing, including saving and displyaing output.
        save_path = os.path.join("../results", f"{args.method}.csv")
        logger.info(f"Save results to {save_path}.")
        write_to_csv(save_path, args, total_accuracy)

    

if __name__ == "__main__":
    main()