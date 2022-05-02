from transformers import AutoTokenizer, AutoConfig, AutoModelForPreTraining, \
    AdamW, get_linear_schedule_with_warmup, \
    TrainingArguments, BeamScorer, Trainer

import torch
from torch.utils.data import Dataset, random_split, DataLoader, \
    RandomSampler, SequentialSampler


def get_tokenizer(special_tokens, model):

    tokenizer = AutoTokenizer.from_pretrained(model) #GPT2Tokenizer

    if special_tokens:
        tokenizer.add_special_tokens(special_tokens)
        print("Special tokens added")
    return tokenizer


def get_model(tokenizer, model, device, special_tokens=None, load_model_path=None):

    #GPT2LMHeadModel
    if special_tokens:
        config = AutoConfig.from_pretrained(model,
                                            bos_token_id=tokenizer.bos_token_id,
                                            eos_token_id=tokenizer.eos_token_id,
                                            sep_token_id=tokenizer.sep_token_id,
                                            pad_token_id=tokenizer.pad_token_id,
                                            output_hidden_states=False)
    else:
        config = AutoConfig.from_pretrained(model,
                                            pad_token_id=tokenizer.eos_token_id,
                                            output_hidden_states=False)

    #----------------------------------------------------------------#

    model = AutoModelForPreTraining.from_pretrained(model, config=config)

    if special_tokens:
        #Special tokens added, model needs to be resized accordingly
        model.resize_token_embeddings(len(tokenizer))

    if load_model_path:
        model.load_state_dict(torch.load(load_model_path))

    model.to(device)
    return model