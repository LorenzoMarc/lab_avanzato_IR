import os
import io
import requests
import numpy as np
import pandas as pd
import re
import zipfile
import time
import csv
import requests
import datetime
from itertools import compress
import argparse

# from collections import Counter, defaultdict
# from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

from transformers import AutoTokenizer, AutoConfig, AutoModelForPreTraining, \
    AdamW, get_linear_schedule_with_warmup, \
    TrainingArguments, BeamScorer, Trainer

import torch
from torch.utils.data import Dataset, random_split, DataLoader, \
    RandomSampler, SequentialSampler

# from IPython.display import clear_output
import utils
import NewsDataset as ds
import model_gpt as md


def run(args):
    DEBUG = args.debug
    TRAIN = args.train
    LOAD_TRAINED = args.load_model
    INPUT_DIR = args.data_path
    USE_APEX = args.apex
    APEX_OPT_LEVEL = 'O1'
    MODEL = args.model  # {gpt2, gpt2-medium, gpt2-large, gpt2-xl}
    UNFREEZE_LAST_N = args.unfreeeze_layers  # The last N layers to unfreeze for training
    SPECIAL_TOKENS = {"bos_token": "<|BOS|>",
                      "eos_token": "<|EOS|>",
                      "unk_token": "<|UNK|>",
                      "pad_token": "<|PAD|>",
                      "sep_token": "<|SEP|>"}
    MAXLEN = args.max_length_token  # {768, 1024, 1280, 1600}
    TRAIN_SIZE = args.train_size
    if USE_APEX:
        TRAIN_BATCHSIZE = 4
        BATCH_UPDATE = 16
    else:
        TRAIN_BATCHSIZE = 2
        BATCH_UPDATE = 32
    EPOCHS = args.epochs
    LR = args.learning_rate
    EPS = args.eps
    WARMUP_STEPS = args.warmup_steps
    SEED = args.seed


    utils.seed_everything(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    if TRAIN:
        data = utils.data_preprocessing(INPUT_DIR)
        tokenizer = md.get_tokenier(SPECIAL_TOKENS, MODEL)
        model = md.get_model(tokenizer,MODEL, device,
                          special_tokens=SPECIAL_TOKENS,
                          load_model_path=LOAD_TRAINED
                          )
        # - Freeze selective layers:
        # - Freeze all layers except last n:
        for parameter in model.parameters():
            parameter.requires_grad = False

        for i, m in enumerate(model.transformer.h):
            # Only un-freeze the last n transformer blocks
            if i + 1 > 12 - UNFREEZE_LAST_N:
                for parameter in m.parameters():
                    parameter.requires_grad = True

        for parameter in model.transformer.ln_f.parameters():
            parameter.requires_grad = True

        for parameter in model.lm_head.parameters():
            parameter.requires_grad = True

        train_data, val_data = ds.split_data(data, TRAIN_SIZE)

        train_dataset = ds.NewsDataset(train_data, tokenizer, MAXLEN)
        val_dataset = ds.NewsDataset(val_data, tokenizer, MAXLEN, randomize=False)

        f'There are {len(train_dataset) :,} samples for training, and {len(val_dataset) :,} samples for validation testing'
        # % % time

        training_args = TrainingArguments(
            output_dir="./",
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=TRAIN_BATCHSIZE,
            per_device_eval_batch_size=TRAIN_BATCHSIZE,
            gradient_accumulation_steps=BATCH_UPDATE,
            evaluation_strategy="epoch",
            # fp16=True,
            # fp16_opt_level=APEX_OPT_LEVEL,
            warmup_steps=WARMUP_STEPS,
            learning_rate=LR,
            adam_epsilon=EPS,
            weight_decay=0.01,
            save_total_limit=1,
            load_best_model_at_end=False,
        )

        # ---------------------------------------------------#
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer
        )

        # ---------------------------------------------------#
        trainer.train()
        trainer.save_model()

    else:

        keywords = ['russia', 'italy', 'berlusconi', 'trump', 'gas', 'sanctions']
        kw = ','.join(keywords)

        prompt = SPECIAL_TOKENS['bos_token'] + kw + SPECIAL_TOKENS['sep_token']

        generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
        device = torch.device("cuda")
        generated = generated.to(device)
        model = md.get_model(tokenizer,MODEL, device,
                             special_tokens=SPECIAL_TOKENS,
                             load_model_path=LOAD_TRAINED
                             )
        model.eval();

        # Top-p (nucleus) text generation (10 samples):
        sample_outputs = model.generate(generated,
                                        do_sample=True,
                                        min_length=50,
                                        max_length=MAXLEN,
                                        top_k=30,
                                        top_p=0.7,
                                        temperature=0.9,
                                        repetition_penalty=2.0,
                                        num_return_sequences=10
                                        )

        for i, sample_output in enumerate(sample_outputs):
            text = tokenizer.decode(sample_output, skip_special_tokens=True)
            a = len(','.join(keywords))
            print("{}: {}\n\n".format(i + 1, text[a:]))

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--keywords', type=str, required=False)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--train', type=bool, required=False, default=True)
    parser.add_argument('--apex', type=bool, default=False)
    parser.add_argument('-samples', '--num_samples', type=int, default=1)
    parser.add_argument('-wu', '--warmup_steps', type=float, default=1e2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('-nfrz_lyrs', '--unfreeeze_layers', type=int, default=6, choices=[0, 2, 4, 6, 8])
    parser.add_argument('-lr', '--learning_rate', type=float, default=5e-4)
    parser.add_argument('-eps', '--eps', type=float, default=1e-8)
    parser.add_argument('-tr_size', '--train_size', type=float, default=0.9)
    parser.add_argument('-max_l', '--max_length_token', type=int, default=768,
                        choices=[768, 1024, 1280, 1600])
    parser.add_argument('-model', '--model', type=str, default='gpt2',
                        choices=['gpt2', 'gpt2-medium', ' gpt2-large', 'gpt2-xl'])
    parser.add_argument('--save_model', type=bool, default=True)
    parser.add_argument('--load_model', type=str, default=None)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    run(args)
