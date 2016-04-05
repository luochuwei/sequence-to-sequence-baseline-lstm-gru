#-*- coding:utf-8 -*-
####################################################
#
#    Author: Chuwei Luo
#    Email: luochuwei@gmail.com
#    Date: 21/02/2016
#    Usage: new Main (in case of the out of memory)
#
####################################################

import time
import gc
import sys
import numpy as np
import theano
import theano.tensor as T
from utils_pg import *
from focus_of_attention_NN import *
import get_data

e = 0.01   #error
lr = 0.2
drop_rate = 0.
batch_size = 200
read_data_batch = 200
# full_data_len = 190363

full_data_len =200
hidden_size = [500]
# try: gru, lstm
cell = "gru"
# try: sgd, momentum, rmsprop, adagrad, adadelta, adam
optimizer = "adadelta" 


x_path = "data/200.post"
y_path = "data/200.response"



# batch_size = 10
# read_data_batch = 10
# # full_data_len = 190363

# full_data_len = 10

# x_path = "data/toy2.txt"
# y_path = "data/toy3.txt"
# test_path = "data/toy2.txt"


threshold = 0


# _, _, i2w, w2i, tf, _ = get_data.processing(x_path, y_path, threshold, 0, 1, 1)
X_seqs, y_seqs, i2w, w2i, tf, data_x_y = get_data.processing(x_path, y_path, threshold, 0, 200, batch_size)
# test_data_x_y = get_data.test_processing(test_path, i2w, w2i, batch_size)

print "#dic = " + str(len(w2i))
# print "unknown = " + str(tf["<UNknown>"])

dim_x = len(w2i)
dim_y = len(w2i)
num_sents = batch_size

print "#features = ", dim_x, "#labels = ", dim_y

print "compiling..."
model = FANN(dim_x, dim_y, hidden_size, cell, optimizer, drop_rate, num_sents)
load_model("data/GRU-200_best.model", model)

print "predict..."






sents = model.predict(data_x_y[0][0], data_x_y[0][1],data_x_y[0][3], batch_size)

get_data.print_sentence(sents[0], dim_y, i2w)


