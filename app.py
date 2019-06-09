# from flask import Flask, render_template, url_for, request
from pathlib import Path
# import requests
import warnings

# warnings.filterwarnings('ignore')
import pandas as pd
import string
import random
import pandas as pd
import numpy as np
import gc
import os
import re
import keras
import pickle

from keras.layers import Dense, Input, Embedding, Dropout, Activation, Conv1D, GlobalMaxPool1D, GlobalMaxPooling1D, \
    concatenate, SpatialDropout1D
from keras.models import Model, load_model, model_from_json
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras import backend as K
from keras.engine import InputSpec, Layer
from keras.optimizers import Adam, RMSprop
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras import callbacks


def clean(text):
    x = text.lower()
    x = x.replace(r'+ve', ' positive ')
    x = x.replace(r'+', ' positive ')
    x = re.sub(r"\b([l][.])", "left ", x)
    x = re.sub(r"\b([r][.])", "right ", x)
    x = re.sub(r"(?<=\d)(st|nd|rd|th)\b", '', x)
    x = ((''.join('#' + i + '#' if i.isdigit() else i for i in x)).replace('#/#', '<>')).replace('#', '')
    x = x.replace(r" rt l", " right lower")
    x = x.replace(r" right l", " right lower")
    x = x.replace(r" le l", " left lower")
    x = x.replace(r" lft l", " left lower")
    x = x.replace(r"\n", r" ")
    x = x.replace(r"\t", r" ")
    x = x.replace(r"\b", r" ")
    x = x.replace(r"\r", r" ")
    x = re.sub(r"\s+", r" ", x)
    x = re.sub(r'[\?\.\!\,\=]+(?=[\?\.\!\,\=])', '', x)
    toks = re.split(' |;|,|\*|\n|[(]|[)]|/|[+]|:|-', x)
    ltok = [mispell_dict[tok] if mispell_dict.get(tok) is not None else tok for tok in toks]
    x = [word_index[k] if word_index.get(k) is not None else 1 for k in ltok]
    x = x[:7]
    x = np.array([[0] * (7 - len(x)) + x if len(x) < 7 else x])
    return x


# KB
def load_KB():
    mispell_dict = np.load(os.getcwd() + '/datafile/mispell_dict.npy', allow_pickle=True).item()
    icd_dict = np.load(Path(os.getcwd()) / 'datafile' / 'icd_dict.npy', allow_pickle=True).item()
    word_index = np.load(Path(os.getcwd()) / 'datafile' / 'word_index.npy', allow_pickle=True).item()
    return mispell_dict, icd_dict, word_index


mispell_dict, icd_dict, word_index = load_KB()
itoicd = dict((v, k) for k, v in icd_dict.items())
with open(Path(os.getcwd()) / 'datafile' / 'tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


def get_result(str_text):
    x_test = clean(str_text)
    score = 0
    for i in range(5):
        model_name = 'model' + str(i) + '.json'
        json_file = open(Path(os.getcwd()) / 'models' / model_name, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        modelh_name = 'model' + str(i) + '.h5'
        loaded_model.load_weights(Path(os.getcwd()) / 'models' / modelh_name)
        loaded_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-3), metrics=["accuracy"])
        score = score + loaded_model.predict(x_test, batch_size=1, verbose=1) / 5
        del loaded_model
    icd = itoicd[np.argmax(score)]
    prob = np.max(score)
    if prob < 0.65: icd = "not_found"
    return (icd, prob)

def text_predict(str_text):
    x_test = clean(str_text)
    partial_scores = []
    score = 0
    for i in range(5):
        model_name = 'model' + str(i) + '.json'
        json_file = open(Path(os.getcwd()) / 'models' / model_name, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        modelh_name = 'model' + str(i) + '.h5'
        loaded_model.load_weights(Path(os.getcwd()) / 'models' / modelh_name)
        loaded_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-3), metrics=["accuracy"])
        #score = loaded_model.predict(x_test, batch_size=1, verbose=1) / 5
        score = loaded_model.predict(x_test, batch_size=1, verbose=1)
        p_icd = itoicd[np.argmax(score)]
        partial_scores.append(list([int(p_icd), float(np.max(score))]))
        del loaded_model

    return partial_scores
