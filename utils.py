import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import Transformer
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from transformers import (
    BertTokenizerFast,
    AutoModelForMaskedLM,
    AutoModelForTokenClassification,
)

DEVICE = torch.device('cpu')

# load bert model
tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
bert_model = AutoModelForMaskedLM.from_pretrained(
    'ckiplab/bert-base-chinese', output_hidden_states=True).to(DEVICE)

# load model for intent detection
ID_model_path = "model\\ID_model"
# load model
ID_classifier = ID_module().to(DEVICE)
ID_classifier.load_state_dict(torch.load(ID_model_path))

# load model for slot filling
SF_model_path = "C:\\Users\\leosh\\OneDrive\\Desktop\\weights-improvement-19.hdf5"
SF_model = keras.models.load_model(SF_model_path)


def get_representation(output):
    """Get the hidden representations from bert layers."""

    # shape: (seq_len, vocab_size)
    hidden_states = output[1]

    token_embeddings = torch.stack(hidden_states, dim=0)
    # remove dimension 1 (batches)
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    # swap dimension 0 and 1
    token_embeddings = token_embeddings.permute(1, 0, 2)
    # the last hidden layer output (2+seq_len, 768)
    hidden_states = [token[-1] for token in token_embeddings]

    return hidden_states


def get_intent(sentence, ID_model):
    # predict by ID_model


def predict_sf(X, SF_model):
    """Predict slots with ckip bert representation and BiLSTM."""

    X = " ".join(X)
    X_encoding = tokenizer.encode_plus(
        X, add_special_tokens=True, return_tensors='pt')
    X_ids = X_encoding['input_ids'].to(DEVICE)
    with torch.no_grad():
        output = bert_model(X_ids)

    # get the 768-d representation of tokens except [CLS] and [SEP]
    LSTM_input = get_representation(output)[1:-1]
    LSTM_input = [np.array(i) for i in LSTM_input]
    tt = []
    tt.append(np.array(LSTM_input))
    output_label = SF_model.predict_classes(np.array(tt))

    return output_label[0]


def extract(intent, SF_list, sentence):
    """Extract the money amount and item from the input sentence."""

    sen = "".join(sentence.split())
    item = ""
    money = ""

    # extract money amount
    money_start_idx = 0
    money_end_idx = 0

    for i, s in enumerate(SF_list):
        if (s == 2) or (s == 3):
            item += sen[i]

        if (s == 4):
            money_start_idx = i
        if (s == 5 and sen[i].isdigit()):
            money_end_idx = i

    if money_start_idx and money_end_idx:
        money = intent + sen[money_start_idx:money_end_idx+1]
    elif money_start_idx:
        money = intent + sen[money_start_idx]

    # item: str, money: str
    return item, money
