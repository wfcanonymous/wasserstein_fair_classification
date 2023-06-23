import sys

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, BertModel, BertTokenizer
from datasets import load_dataset
import pandas as pd


def set_processor(use_gpu=True):
    if use_gpu:
        if torch.cuda.is_available():
            # Tell PyTorch to use the GPU.
            device = torch.device("cuda")
            print('We will use the GPU:', torch.cuda.get_device_name(0))
        else:
            print('No GPU available, using the CPU instead.')
            device = torch.device("cpu")
            use_gpu = False
    else:
        device = torch.device("cpu")
        use_gpu = False
    return device, use_gpu


def batch_encode_ant(x, tokenizer, max_seq_len=128, batch_size=256, return_mask=False):
    if type(x) is not list:
        x = x.tolist()

    if return_mask:
        tok = []
        mask = []
        for i in range(0, len(x), batch_size):
            bob = tokenizer(x[i: i + batch_size], return_tensors="pt", padding="max_length", max_length=max_seq_len,
                            truncation=True)
            tok.append(bob["input_ids"])
            mask.append(bob["attention_mask"])
        tok = torch.cat(tok, 0)
        mask = torch.cat(mask, 0)
        return tok, mask
    else:
        tok = []
        for i in range(0, len(x), batch_size):
            bob = tokenizer(x[i: i + batch_size], return_tensors="pt", padding="max_length", max_length=max_seq_len,
                            truncation=True)
            tok.append(bob["input_ids"])
        tok = torch.cat(tok, 0)
        return tok


def load_dataframe(dataset_name):
    print(dataset_name)
    if dataset_name == "EEC":
        persons = ["she", "her", "he", "him", "this woman", "this man", "this girl", "this boy", "my sister",
                   "my brother", "my daughter", "my son", "my wife", "my husband", "my girlfriend", "my boyfriend",
                   "my mother", "my father", "my aunt", "my uncle", "my mom", "my dad"]
        dataset = load_dataset("peixian/equity_evaluation_corpus")
        dataset = pd.DataFrame({'text': dataset['train']['sentence'],
                                'gender': dataset['train']['gender'],
                                'person': dataset['train']['person']
                                })
        dataset = dataset[dataset['person'].isin(persons)].sample(frac=1).reset_index()
        dataset = dataset.drop(columns=['person', 'index'])
        dataset['gender'] = dataset['gender'].map({'female': 1, 'male': 0})
    elif dataset_name == "dv2_story":
        dataset = pd.read_csv("data/dv2_story_generations.csv")
        dataset = dataset[['text', 'gender']]
        dataset['gender'] = dataset['gender'].map({'W': 1, 'M': 0})
        dataset['text'] = dataset['text'].str.lstrip('\n\n')
    else:
        print('name error')
        sys.exit(1)
    return dataset

# https://github.com/shauli-ravfogel/nullspace_projection/blob/master/src/data/encode_bert_states.py
def load_lm():
    """
    load bert's language model
    :return: the model and its corresponding tokenizer
    """
    model_class, tokenizer_class, pretrained_weights = (BertModel, BertTokenizer, 'bert-base-uncased')
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)
    return model, tokenizer


def tokenize(tokenizer, data):
    """
    Iterate over the data and tokenize it. Sequences longer than 512 tokens are trimmed.
    :param tokenizer: tokenizer to use for tokenization
    :param data: data to tokenize
    :return: a list of the entire tokenized data
    """
    tokenized_data = []
    for row in tqdm(data):
        tokens = tokenizer.encode(row, add_special_tokens=True)
        # keeping a maximum length of bert tokens: 512
        tokenized_data.append(tokens[:512])
    return tokenized_data


# https://github.com/shauli-ravfogel/nullspace_projection/blob/master/src/data/encode_bert_states.py
def encode_text(model, data):
    """
    encode the text
    :param model: encoding model
    :param data: data
    :return: two numpy matrices of the data:
                first: average of all tokens in each sentence
                second: cls token of each sentence
    """
    all_data_cls = []
    all_data_avg = []
    batch = []
    for row in tqdm(data):
        batch.append(row)
        input_ids = torch.tensor(batch)
        with torch.no_grad():
            last_hidden_states = model(input_ids)[0]
            all_data_avg.append(last_hidden_states.squeeze(0).mean(dim=0).numpy())
            all_data_cls.append(last_hidden_states.squeeze(0)[0].numpy())
        batch = []
    return np.array(all_data_avg), np.array(all_data_cls)


def load_train_dataloaders(dataset_name, batch_size=32, model_name="bert-base-uncased"):
    model, tokenizer = load_lm()

    train_set = load_dataframe(dataset_name)

    # Tokenize text
    tokens = tokenize(tokenizer, train_set['text'])
    # Encode text with BERT as in Null it Out Ravfogel et al. 2020
    avg_data, _ = encode_text(model, tokens)
    labels = torch.tensor(train_set['gender'].astype(np.int32))

    train_data = TensorDataset(torch.tensor(avg_data), labels)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    return train_loader
