import argparse
import os

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
)

# init a args parser
def parse_args():
    parser = argparse.ArgumentParser(description="Language model downloaders.")
    
    parser.add_argument(
        "--model_family",
        type=str,
        choices=["GPT2", "T5", "FlanT5"],
        default=None,
        help="The moddel family, as checkpoints under the same model family use same codes to download."
        )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="The checkpoint name under a model family, e.g. gpt2, gpt2-medium, gpt2-large, gpt2-xl."
    )
    

    parser.add_argument(
        "--output_dir",
        type=str,
        default=f"./models",
    )
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    if args.model_family == "GPT2":
        tokenizer_func = AutoTokenizer
        model_func = AutoModelForCausalLM
    else:
        print("not implemented")
        return

    # TBD: check the checkpoint are valid
    print("Checkpoint is valid.")

    # download the model
    tokneizer = tokenizer_func.from_pretrained(args.checkpoint)
    model = model_func.from_pretrained(args.checkpoint)

    # save the model
    save_dir = os.path.join(args.output_dir, args.model_family, args.checkpoint)
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    model.save_pretrained(save_dir)
    tokneizer.save_pretrained(save_dir)

if __name__ == "__main__":
    main()