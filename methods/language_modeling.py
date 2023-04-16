# a framework for inference on multiple choice tasks.
import argparse
import logging
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import(
    AutoTokenizer, 
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)
from datasets import Dataset


from utils.data import(
    copa_loader,
    winogrande_loader,
)

logger = logging.getLogger(__name__)

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
        choices=["copa", "winogrande"],
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

def main():
    # import pdb; pdb.set_trace()

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
    model.eval()

    # step 4: load and preprocess data.
    logger.info(f"Load data: {args.data}.")
    # prepcrocess data: using specific function for each dataset.
    if args.data == "copa":
        file_path = os.path.join("../data", args.data, "copa-dev.xml")
        data_loader = copa_loader
    elif args.data == "winogrande":
        file_path = os.path.join("../data", args.data, "dev.jsonl")
        data_loader = winogrande_loader
    
    dev_data = data_loader(file_path)
    # next steps: 
    # 1. preprocess the data with tokenizer
    # 2. use datacollator to form batches


    # consider using a dataloader here.
    # https://huggingface.co/docs/datasets/use_with_pytorch
    # dataset = Dataset.from_list(dev_data).with_format("torch")
    # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    # for batch in dataloader:
    #     print(batch)
    #     break

    # step 5: inference on data, and compute accuracy.
    logger.info(f"Start inference on {args.data} using {args.model_family} model: {args.checkpoint}.")
    # what are the arguments needed for this step? 
    


    # step 6: some postprocessing, including saving and displyaing output.
    pass

if __name__ == "__main__":
    main()