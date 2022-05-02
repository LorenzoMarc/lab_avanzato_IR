# -*- coding: utf-8 -*-
import argparse
from os import path
import pandas as pd
from tabulate import tabulate
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
def run(args):
    OUTPUT = args.output
    masked = args.masked_set
    original = args.original_set
    # load into a data frame
    df_masked = pd.read_csv(masked)
    df_ori = pd.read_csv(original)
    length = len(df_masked)
    print(length)

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    # if (path.exists(model_dir)):
    #     model = BartForConditionalGeneration.from_pretrained(model_dir)
    # else:
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    with open(OUTPUT, 'a', ) as f:

        for num, samp in enumerate(range(length-1)):
            f.write("\n\n-----####"+str(num+1)+"######--------\n")
            print("-----####"+str(num+1)+"######--------")

            stmt = df_ori.iloc[samp]['statement']
            f.write("-----####ORIGINAL statement: \n" + stmt+"\n")
            print("ORIGINAL statement: \n" + stmt)
            msk = df_masked.iloc[samp]['statement']

            f.write("MASKED statement: \n" + msk+"\n")
            print("MASKED statement: \n" + msk)

            batch = tokenizer(msk, return_tensors="pt")
            generated_ids = model.generate(batch["input_ids"],
                                           max_new_tokens=35,# or this,not both: max_length=300,
                                           # do_sample=True,
                                           top_k=5,
                                           top_p=0.9,
                                           # temperature=0.9,
                                           # num_beams=6,
                                           no_repeat_ngram_size=3,
                                           num_return_sequences=1,
                                           # repetition_penalty=1.3,
                                           min_length=50
                                           # early_stopping=True
                                           )
            res = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            print("GENERATED statement: \n" + res[s] for s in range(len(res)))
            f.write("GENERATED statement: \n" )
            for s in range(len(res)):
                f.write(str(res[s]) + "\n")
    f.close()

    print("CHECK your file " + str(OUTPUT))

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mask', '--masked_set', type=str, default='masked_samples.csv')
    parser.add_argument('-ori', '--original_set', type=str, default='samples_statements.csv')
    parser.add_argument('-out', '--output', type=str, default='predicted_multi_masked_token.txt')
    parser.add_argument('-data', '--dataset', type=str, default='../data/Politifact_20211230.csv')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    run(args)
