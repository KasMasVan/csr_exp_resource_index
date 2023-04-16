# a framework for inference on multiple choice tasks.
import argparse
import logging
import os
import random
import sys

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
import evaluate


from utils.data import(
    copa_loader,
    winogrande_loader,
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
        dev_data = copa_loader(file_path)
        dataset = Dataset.from_list(dev_data).with_format("torch")
    elif args.data == "winogrande":
        file_path = os.path.join("../data", args.data, "dev.jsonl")
        data_loader = winogrande_loader
    
    logger.info(f"Preprocess data: {args.data}.")
    fn_kwargs = {"ending_names": ending_names, 
                 "header_name": header_name, 
                 "tokenizer": tokenizer}
    tokenized_dataset = dataset.map(preprocess_function, fn_kwargs=fn_kwargs, batched=True)
    
    eval_dataloader = DataLoader(tokenized_dataset, batch_size=args.batch_size, shuffle=False)

    # step 5: (evaluation) inference on data, and compute accuracy.
    logger.info(f"Start inference on {args.data} using {args.model_family} model: {args.checkpoint}.")
    metric = evaluate.load("accuracy")
    model.eval()
    for batch in eval_dataloader:
        # need to flatten
        
        header_input_ids = batch["header_input_ids"].view(-1, batch["header_input_ids"].shape[-1]).to(device)
        ending_input_ids = batch["ending_input_ids"].view(-1, batch["ending_input_ids"].shape[-1]).to(device)
        outputs = model(input_ids = header_input_ids, labels = ending_input_ids)
        loss, logits = outputs.loss, outputs.logits
        logits = logits.view(-1, logits.shape[-1])
        loss_none = F.cross_entropy(logits, ending_input_ids.view(-1), reduction="none")
        loss_mean = F.cross_entropy(logits, ending_input_ids.view(-1), reduction="mean")
        print(loss)

        break
        # predictions = torch.argmax(logits, dim=-1)
        # metric.add_batch(predictions=predictions, references=batch["labels"])
    


    # step 6: some postprocessing, including saving and displyaing output.
    pass

if __name__ == "__main__":
    main()