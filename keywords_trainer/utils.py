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
import random
import torch
import math

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def read_keywords():
    keywords = dict()
    key_file = os.path.join("data","keywords.csv")
    with open(key_file, newline='', encoding="utf8") as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            keywords[int(row[0])] = row[1:]
    print(f"Number of entries in keywords: {len(keywords) :,}")
    return keywords


def mean_stmnt(input_path):
    df = pd.read_csv(input_path)
    df = df['statement']
    all = len(df)
    elements=0
    for s in df:
        elements += len(s)
    mean = math.ceil(elements/all)
    return mean
def data_preprocessing(input_path):
    df = pd.read_csv(input_path)
    print(f"df size: {len(df) :,}")
    print(df.head(10))

    keywords = read_keywords()
    df = df['statement']
    data = dict()

    t0 = time.time()

    for i, text in enumerate(df):
        # id, text
        id = i
        data[id] = [text]

    all_keywords = set()
    for k, v in keywords.items():
        for w in v:
            all_keywords.add(w)
    for k in data.keys():
        # data[id] è lo statement con indice id che DEVE essere uguale a keywords id che mantiene le keywords per quello statement di indice id
        data[k].append(keywords[k])
    print(f"Number of unique keywords: {len(all_keywords) :,}")
    return data