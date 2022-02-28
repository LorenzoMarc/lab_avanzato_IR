
import torch
from torch.utils.data import Dataset, random_split, DataLoader, \
                             RandomSampler, SequentialSampler
import random

class NewsDataset(Dataset):

    def __init__(self, data_input, tokenizer, LENGTH, randomize=True):

        statement, keywords = [], []
        for k, v in data_input.items():
            statement.append(v[0])
            keywords.append(v[1])
        self.length = LENGTH
        self.randomize = randomize
        self.tokenizer = tokenizer  # the gpt2 tokenizer we instantiated
        self.statement = statement
        self.keywords = keywords

    def __len__(self):
        return len(self.statement)

    def __getitem__(self, i):
        keywords = self.keywords[i].copy()
        # print("KEYWORDS #######" + str(type(keywords)))
        N = len(keywords)

        # random sampling and shuffle
        if self.randomize:
            M = random.choice(range(N + 1))
            keywords = keywords[:M]
            # print("KEYWORDS RANDOMIZE#######" + str(type(keywords)))
            random.shuffle(keywords)

        # random.shuffle(keywords)#self.join_keywords(keywords, self.randomize)
        # print('sono nell\'if RANDOM SHUFFLE' + str(type(keywords)))
        # print('sono nell\'else' + str(type(kw)))
        
        SPECIAL_TOKENS = {"bos_token": "<|BOS|>",
                          "eos_token": "<|EOS|>",
                          "unk_token": "<|UNK|>",
                          "pad_token": "<|PAD|>",
                          "sep_token": "<|SEP|>"}

        keywords_string = ','.join([str(item) for item in keywords])

        input = SPECIAL_TOKENS['bos_token'] + keywords_string + SPECIAL_TOKENS['sep_token'] + self.statement[i] + \
                SPECIAL_TOKENS['eos_token']

        maxlen = self.length
        encodings_dict = self.tokenizer(input,
                                   truncation=True,
                                   max_length=maxlen,
                                   padding="max_length")

        input_ids = encodings_dict['input_ids']
        attention_mask = encodings_dict['attention_mask']

        return {'label': torch.tensor(input_ids),
                'input_ids': torch.tensor(input_ids),
                'attention_mask': torch.tensor(attention_mask)}

def split_data(data, S):
    # Shuffle ids
    ids = list(data.keys())
    random.shuffle(ids)

    # Split into training and validation sets
    train_size = int(S * len(data))
    val_size = int((1-S) * len(data))

    train_ids = ids[:train_size]
    val_ids = ids[-val_size:]

    train_data = dict()
    for id in train_ids:
        train_data[id] = data[id]

    val_data = dict()
    for id in val_ids:
        val_data[id] = data[id]

    return train_data, val_data