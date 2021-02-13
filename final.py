#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.chdir("C:\\Users\\leosh\\")

from utils import *


# In[2]:


import speech_recognition as sr
import pyaudio
import numpy as np
import pandas as pd


# In[3]:


# old  module
import tensorflow as tf  
from tensorflow import keras
from tensorflow.keras import layers 

#new  module
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import Transformer
import torch.nn as nn
import torch.nn.functional as F
import os
import torch.optim as optim
import random


# In[4]:


def Voice_To_Text():
    #entryVariable.delete(0, 'end')
    #label1.config(text='Lets talk')
    final_text = []
    r = sr.Recognizer()
    with sr.Microphone() as source: 
        entryVariable.delete(0, 'end')
        label1.config(text='Lets talk')
        print("請開始說話:")                     # print 一個提示 提醒你可以講話了
        r.adjust_for_ambient_noise(source)     # 函數調整麥克風的噪音:
        audio = r.listen(source)
    try:
        Text = r.recognize_google(audio, language="zh-TW")
    except sr.UnknownValueError:
        Text = "無法翻譯"
    except sr.RequestError as e:
        Text = "無法翻譯{0}".format(e)
    
    final_text.extend(list(Text))  #output sentence list
    entryVariable.insert(0,final_text)
    label1.config(text='')
    print(final_text)
    return final_text


# In[5]:


def extract(intent, sf_list, sentence):
    sen = "".join(sentence.split())
    item = ""
    money = ""
    # extract item
    # sf_map = map(index_item, list(sf_list))
    # item_mask = list(sf_map)
    # item = "".join(list(compress(sen, item_mask)))

    # extract money amount
    money_start_idx = 0
    money_end_idx = 0

    for i, s in enumerate(sf_list):
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


# In[6]:


def confirm():
    sen = entryVariable.get()
    print(sen)
    c = ckip_BERT_LSTM(sen,"C:\\Users\\leosh\\OneDrive\\Desktop\\weights-improvement-19.hdf5")
    i = get_intent(sen)
    
    item,money = extract(i,c,sen)
    text = label4.cget("text") + item + "\n"
    label4.configure(text=text)
    text = label5.cget("text") + money + "\n"
    label5.configure(text=text)


# In[7]:


def clean():
    label4.configure(text="")
    label5.configure(text="")


# In[8]:


network = keras.models.load_model("C:\\Users\\leosh\\OneDrive\\Desktop\\weights-improvement-19.hdf5")


# In[9]:


from transformers import (
   BertTokenizerFast,
   AutoModelForMaskedLM,
   AutoModelForTokenClassification,
)

DEVICE = torch.device('cpu')

# language model
tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
model = AutoModelForMaskedLM.from_pretrained('ckiplab/bert-base-chinese', output_hidden_states=True).to(DEVICE)


# In[10]:


def ckip_BERT_LSTM(X,filename):
  #load  model
  network = keras.models.load_model(filename)
  X = " ".join(X)
  X_encoding = tokenizer.encode_plus(X, add_special_tokens=True, return_tensors='pt')
  X_ids = X_encoding['input_ids'].to(DEVICE)
  with torch.no_grad():
      output = model(X_ids)
  # get the 768-d representation of other tokens than [CLS]
  LSTM_input = get_representation(output)[1:-1]
  LSTM_input = [np.array(i) for i in LSTM_input]
  tt=[]
  tt.append(np.array(LSTM_input))
  output_label = network.predict_classes(np.array(tt))
  
  return output_label[0]


# In[ ]:





# In[12]:


from tkinter import *
window = Tk()
window.geometry("350x200")

label1 = Label(window, text='')
btn1 = Button(window, text = "麥克風", bg = "yellow", command=Voice_To_Text)
entryVariable = Entry(window, text="這是文字方塊",width=30)
btn2 = Button(window, text = "確定", bg = "yellow", command=confirm)
label2 = Label(window, text='項目')
label3 = Label(window, text='收支出  ')
labele = Label(window, text='')
label4 = Label(window, text='')#item
label5 = Label(window, text='')#price
btn3 = Button(window, text = "清空", bg = "yellow", command=clean)

label1.grid(row = 0, column = 1)
btn1.grid(row = 1, column = 0)
entryVariable.grid(row = 1, column = 1)
btn2.grid(row = 1, column = 2)
labele.grid(row = 2, column = 0)
label2.grid(row = 3, column = 0)
label3.grid(row = 3, column = 2)
label4.grid(row = 4, column = 0)
label5.grid(row = 4, column = 2)
btn3.grid(row=1,column=3)
#window.maxsize(400,400) #int
window.mainloop()


# In[ ]:




