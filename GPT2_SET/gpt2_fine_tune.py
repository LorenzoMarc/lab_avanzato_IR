# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
import os
import time
import datetime
from os import path
import pandas as pd
import seaborn as sns
import numpy as np
import random

import matplotlib.pyplot as plt
# % matplotlib inline

import torch
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
import argparse

from transformers import  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from transformers import AdamW, get_linear_schedule_with_warmup

import nltk

nltk.download('punkt')

def run(args):

    filename = args.dataset
    # load into a data frame
    SET = args.set
    NUM_SAMPLES = args.num_samples
    OUTPUT = args.output_dir
    df = pd.read_csv(filename)

    df.dropna(inplace=True)  # remove NA values
    df = df[['statement', 'target']]
    if (SET==0):
        df = df[df['target'].isin(['false', 'pants-fire', 'barely-true'])]

    else:
        df = df[df['target'].isin(['true', 'mostly-true', 'half-true'])]


    statements = df.statement.copy()  # just use the main bio text in this example
    print(statements)

    # STATS
    doc_lengths = []

    for s in statements:
        # get rough token count distribution
        tokens = nltk.word_tokenize(s)

        doc_lengths.append(len(tokens))

    doc_lengths = np.array(doc_lengths)
    max_len = np.amax(doc_lengths)
    avg_len = np.average(doc_lengths)
    sns.distplot(doc_lengths).set(title='Tokens Length Distribution')
    plt.text(78, 0.005, "max length:\n" + str(max_len), horizontalalignment='left', size='small', color='black',
             weight='semibold')
    plt.savefig('doc_distr_test30.png')
    plt.close()

    # model config
    output_dir = OUTPUT
    print(type(output_dir))

    if (path.exists(output_dir)):
        tokenizer = GPT2Tokenizer.from_pretrained(output_dir, bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
        model = GPT2LMHeadModel.from_pretrained(output_dir)
    else:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>',
                                                  pad_token='<|pad|>')  # gpt2-medium
        configuration = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)
        # instantiate the model
        model = GPT2LMHeadModel.from_pretrained("gpt2", config=configuration)

    print("The max model length is {} for this model, although the actual embedding size for GPT small is 768".format(
        tokenizer.model_max_length))
    print("The beginning of sequence token {} token has the id {}".format(
        tokenizer.convert_ids_to_tokens(tokenizer.bos_token_id), tokenizer.bos_token_id))
    print("The end of sequence token {} has the id {}".format(tokenizer.convert_ids_to_tokens(tokenizer.eos_token_id),
                                                              tokenizer.eos_token_id))
    print("The padding token {} has the id {}".format(tokenizer.convert_ids_to_tokens(tokenizer.pad_token_id),
                                                      tokenizer.pad_token_id))
    # with 4 raise memory error
    batch_size = 2

    #DATASET collator
    class GPT2Dataset(Dataset):

        def __init__(self, txt_list, tokenizer, max_length=768):
            self.tokenizer = tokenizer
            self.input_ids = []
            self.attn_masks = []

            for txt in txt_list:
                encodings_dict = tokenizer('<|startoftext|>' + txt + '<|endoftext|>', truncation=True,
                                           max_length=max_length, padding="max_length")

                self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
                self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            return self.input_ids[idx], self.attn_masks[idx]


    dataset = GPT2Dataset(statements, tokenizer, max_length=768)

    # Split into training and validation sets
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print('{:>5,} training samples'.format(train_size))
    print('{:>5,} validation samples'.format(val_size))

    # Create the DataLoaders for our training and validation datasets.
    # We'll take training samples in random order.
    train_dataloader = DataLoader(
        train_dataset,  # The training samples.
        sampler=RandomSampler(train_dataset),  # Select batches randomly
        batch_size=batch_size  # Trains with this batch size.
    )

    # For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader = DataLoader(
        val_dataset,  # The validation samples.
        sampler=SequentialSampler(val_dataset),  # Pull out batches sequentially.
        batch_size=batch_size  # Evaluate with this batch size.
    )

    # this step is necessary because I've added some tokens (bos_token, etc) to the embeddings
    # otherwise the tokenizer and model tensors won't match up
    model.resize_token_embeddings(len(tokenizer))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)

    # Set the seed value all over the place to make this reproducible.
    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    epochs = 5
    learning_rate = 5e-4
    warmup_steps = 1e2
    epsilon = 1e-8

    # this produces sample output every 100 steps
    sample_every = 100

    # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
    optimizer = AdamW(model.parameters(),
                      lr=learning_rate,
                      eps=epsilon
                      )

    # Total number of training steps is [number of batches] x [number of epochs].
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    # This changes the learning rate as the training loop progresses
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)


    def format_time(elapsed):
        return str(datetime.timedelta(seconds=int(round((elapsed)))))


    total_t0 = time.time()

    training_stats = []

    model = model.to(device)

    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        t0 = time.time()

        total_train_loss = 0

        model.train()

        for step, batch in enumerate(train_dataloader):

            b_input_ids = batch[0].to(device)
            b_labels = batch[0].to(device)
            b_masks = batch[1].to(device)

            model.zero_grad()

            outputs = model(b_input_ids,
                            labels=b_labels,
                            attention_mask=b_masks,
                            token_type_ids=None
                            )

            loss = outputs[0]

            batch_loss = loss.item()
            total_train_loss += batch_loss

            # Get sample every x batches.
            if step % sample_every == 0 and not step == 0:

                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}. Loss: {:>5,}.   Elapsed: {:}.'.format(step, len(train_dataloader),
                                                                                         batch_loss, elapsed))

                model.eval()

                sample_outputs = model.generate(
                    bos_token_id=random.randint(1, 30000),
                    do_sample=True,
                    top_k=50,
                    max_length=200,
                    top_p=0.95,
                    num_return_sequences=1
                )
                for i, sample_output in enumerate(sample_outputs):
                    print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))

                model.train()

            loss.backward()

            optimizer.step()

            scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))

        # ========================================
        #               Validation
        # ========================================

        print("")
        print("Running Validation...")

        t0 = time.time()

        model.eval()

        total_eval_loss = 0
        nb_eval_steps = 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            b_input_ids = batch[0].to(device)
            b_labels = batch[0].to(device)
            b_masks = batch[1].to(device)

            with torch.no_grad():
                outputs = model(b_input_ids,
                                attention_mask=b_masks,
                                labels=b_labels)

                loss = outputs[0]

            batch_loss = loss.item()
            total_eval_loss += batch_loss

        avg_val_loss = total_eval_loss / len(validation_dataloader)

        validation_time = format_time(time.time() - t0)

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

    print("")
    print("Training complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

    # Display floats with two decimal places.
    pd.set_option('precision', 2)

    # Create a DataFrame from our training statistics.
    df_stats = pd.DataFrame(data=training_stats)

    # Use the 'epoch' as the row index.
    df_stats = df_stats.set_index('epoch')

    # Display the table.
    print(df_stats)

    # Use plot styling from seaborn.
    sns.set(style='darkgrid')

    # Increase the plot size and font size.
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12, 6)

    # Plot the learning curve.
    plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
    plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validation")

    # Label the plot.
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.xticks([1, 2, 3, 4,5])

    plt.savefig('loss.png')
    plt.close()
    # Get all of the model's parameters as a list of tuples.
    params = list(model.named_parameters())

    print('The GPT-2 model has {:} different named parameters.\n'.format(len(params)))

    print('==== Embedding Layer ====\n')

    for p in params[0:2]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    print('\n==== First Transformer ====\n')

    for p in params[2:14]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    print('\n==== Output Layer ====\n')

    for p in params[-2:]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()

    output_dir = OUTPUT

    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Saving model to %s" % output_dir)

    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    model.eval()

    prompt = "<|startoftext|>"

    generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    generated = generated.to(device)

    print(generated)

    sample_outputs = model.generate(
        generated,
        do_sample=True,
        top_k=50,
        max_length=300,
        top_p=0.95,
        num_return_sequences=NUM_SAMPLES
    )

    with open('generated_statements.txt', 'a') as f:
        for i, sample_output in enumerate(sample_outputs):
            f.write(str(i) + ': ' + tokenizer.decode(sample_output, skip_special_tokens=True) + "\n")
            print("{}: {}\n\n".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
    f.close()

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-set', '--set', type=int, choices=[0,1], default=0)
    parser.add_argument('-n', '--num_samples', type=int, default=10)
    parser.add_argument('-out', '--output_dir', type=str, default='./model_save_test30/')
    parser.add_argument('-data', '--dataset', type=str, default='../data/Politifact_20211230.csv')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    run(args)
