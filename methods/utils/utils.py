import argparse
import csv
import logging
import os
import random
import sys
from tqdm import tqdm

import numpy as np
import torch

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed_all(seed)

def write_to_csv(save_path, args, total_accuracy):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    csv_exists = os.path.isfile(save_path)
    with open(save_path, 'a+', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        if not csv_exists:
            csvwriter.writerow(['model_family', 'checkpoint', 'data', 'batch_size', 'method', "seed", 'accuracy'])
        csvwriter.writerow([args.model_family, args.checkpoint, args.data, args.batch_size, args.method, args.seed, f"{total_accuracy:.4f}"])