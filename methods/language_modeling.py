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
    copa_loader,
    cqa_loader,
    winogrande_loader,
)
from utils.utils import(
    set_seed,
    write_to_csv,
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

    args = parser.parse_args()
    return args

def inference_language_modeling(model, eval_dataloader, device):
    model.eval()
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
        
        # adding this line of code takes me more than an hour.
        # without adding torch.no_grad, GPU usage will muiltply by 4.
        with torch.no_grad():
            outputs = model(input_ids = header_input_ids, labels = ending_input_ids)
        
        _, logits = outputs.loss, outputs.logits
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

def main():
    # import pdb; pdb.set_trace()

    # step 1: argument parser, and logger
    args = parse_args()
    args.method = "language_modeling"
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
                 "tokenizer": tokenizer,}
    tokenized_dataset = dataset.map(preprocess_function, fn_kwargs=fn_kwargs, batched=True, batch_size=args.batch_size)
    
    eval_dataloader = DataLoader(tokenized_dataset, batch_size=args.batch_size, shuffle=False)

    # step 5: (evaluation) inference on data, and compute accuracy.
    logger.info(f"Start inference (method: {args.method}) on {args.data} using {args.model_family} model: {args.checkpoint}.")
    total_accuracy = inference_language_modeling(model, eval_dataloader, device)

 
    # step 6: some postprocessing, including saving and displyaing output.

    save_path = os.path.join("../results", f"{args.method}.csv")
    logger.info(f"Save results to {save_path}.")
    write_to_csv(save_path, args, total_accuracy)

if __name__ == "__main__":
    main()