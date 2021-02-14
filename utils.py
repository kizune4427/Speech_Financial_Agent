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

# model definition
n_classes = 5


class ID_module(nn.Module):
    """Intent detection module."""

    def __init__(self):
        super(ID_module, self).__init__()
        self.linear = nn.Linear(768, 2)

    def forward(self, X):
        X = self.linear(X)
        return X


class SF_module(nn.Module):
    """Slot filling module."""

    def __init__(self):
        super(SF_module, self).__init__()
        self.linear = nn.Linear(768, n_classes)

    def forward(self, X):
        X = self.linear(X)
        return X


# load bert model
tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
bert_model = AutoModelForMaskedLM.from_pretrained(
    'ckiplab/bert-base-chinese', output_hidden_states=True).to(DEVICE)

# load model for intent detection
ID_model_path = "model/ID_model"
# load model
ID_model = ID_module().to(DEVICE)
ID_model.load_state_dict(torch.load(ID_model_path))

# load model for slot filling
SF_BiLSTM_model_path = "model/weights-improvement-19.hdf5"
SF_MLP_model_path = "model/SF_model"


def get_SF_model(model_type):
    """Returns a slot filling model of either using BiLSTM or MLP method."""
    if model_type == "BiLSTM":
        return keras.models.load_model(SF_BiLSTM_model_path)
    elif model_type == "MLP":
        SF_model = SF_module().to(DEVICE)
        SF_model.load_state_dict(torch.load(SF_MLP_model_path))
        return SF_model


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


def get_intent(sentence):
    """Returns the intent (income/expense) of the given sentence."""

    X = " ".join(sentence)
    X_encoding = tokenizer.encode_plus(
        X, add_special_tokens=True, return_tensors='pt')
    X_ids = X_encoding['input_ids'].to(DEVICE)

    with torch.no_grad():
        output = bert_model(X_ids)

    ID_model.eval()
    ID_input = get_representation(output)[0]
    ID_output = ID_model(ID_input)
    result = torch.argmax(ID_output).item()

    return result


def predict_sf(sentence, model_type="MLP"):
    """Predict the slots with ckip bert representation and SF_model.\n
    model_type: BiLSTM, MLP
    """

    X = " ".join(sentence)
    X_encoding = tokenizer.encode_plus(
        X, add_special_tokens=True, return_tensors='pt')
    X_ids = X_encoding['input_ids'].to(DEVICE)
    with torch.no_grad():
        output = bert_model(X_ids)

    # get the 768-d representation of tokens except [CLS] and [SEP]
    SF_input = get_representation(output)[1:-1]
    y_pred = []

    SF_model = None
    if model_type == "BiLSTM":
        SF_model = get_SF_model("BiLSTM")

        LSTM_input = [np.array(i) for i in SF_input]
        y_pred.append(np.array(LSTM_input))
        y_pred = SF_model.predict_classes(np.array(y_pred))
        return y_pred[0]

    elif model_type == "MLP":
        SF_model = get_SF_model("MLP")
        SF_model.eval()

        for p in range(len(X_ids[0]) - 2):
            SF_output = SF_model(SF_input[p]).unsqueeze(0)
            y_pred.append(torch.argmax(SF_output).item())

        return y_pred
    else:
        print("Model type not supported!")
        print('Please choose from either "BiLSTM" or "MLP"')


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
