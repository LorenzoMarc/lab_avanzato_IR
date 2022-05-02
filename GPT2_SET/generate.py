import os
import time
import datetime
from os import path
import pandas as pd
import seaborn as sns
import numpy as np
import random

import matplotlib.pyplot as plt
import argparse
import torch


def run(args):
    NUM_SAMPLES = args.num_statements
    MAX_LENGTH = args.max_length
    FILENAME = args.file_name
    MODEL = args.model_dir

    from transformers import GPT2Tokenizer, GPT2LMHeadModel

    model_dir = MODEL
    if (path.exists(model_dir)):
        tokenizer = GPT2Tokenizer.from_pretrained(model_dir, bos_token='<|startoftext|>', eos_token='<|endoftext|>',
                                                  pad_token='<|pad|>')
        model = GPT2LMHeadModel.from_pretrained(model_dir)
    else:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>',
                                                  pad_token='<|pad|>')
        model = GPT2LMHeadModel.from_pretrained('gpt2')

    # this step is necessary because I've added some tokens (bos_token, etc) to the embeddings
    # otherwise the tokenizer and model tensors won't match up
    model.resize_token_embeddings(len(tokenizer))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    prompt = "<|startoftext|>"

    generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    generated = generated.to(device)

    print(generated)

    sample_outputs = model.generate(
        generated,
        do_sample=True,
        top_k=50,
        max_length=MAX_LENGTH,
        top_p=0.95,
        num_return_sequences=NUM_SAMPLES
    )


    with open(FILENAME + '.txt', 'a') as f:
        for i, sample_output in enumerate(sample_outputs):
            f.write(tokenizer.decode(sample_output, skip_special_tokens=True) + "\n")
            # print("{}: {}\n\n".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
    f.close()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-max_l', '--max_length', type=int, default=150)
    parser.add_argument('-n', '--num_statements', type=int, default=10)
    parser.add_argument('-f', '--file_name', type=str, default='generated_statements')
    parser.add_argument('-model', '--model_dir', type=str, default='model_save_test30')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    run(args)
