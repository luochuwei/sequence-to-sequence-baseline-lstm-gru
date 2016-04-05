#-*- coding:utf-8 -*-
############################################
#
#    Author: Chuwei Luo
#    Email: luochuwei@gmail.com
#    Date: 11/02/2016
#    Usage: Main
#
############################################
import time
import sys
import numpy as np
import theano
import theano.tensor as T
from utils_pg import *
from focus_of_attention_NN import *
import get_data

e = 0.01   #error
lr = 0.5
drop_rate = 0.
batch_size = 1
hidden_size = [500]
# try: gru, lstm
cell = "gru"
# try: sgd, momentum, rmsprop, adagrad, adadelta, adam
optimizer = "adadelta" 


x_path = "data/X.txt"
y_left_path = "data/y_left.txt"
y_right_path = "data/y_right.txt"

threshold = 10

X_seqs, yl_seqs, yr_seqs, i2w, w2i, data_x_yl_yr = get_data.processing(x_path, y_left_path, y_right_path, threshold, batch_size)
dim_x = len(w2i)
dim_y = len(w2i)
num_sents = data_x_yl_yr[0][-1]

print "save data dic..."
save_data_dic("data/i2w.pkl", "data/w2i.pkl", i2w, w2i)

print "#features = ", dim_x, "#labels = ", dim_y

print "compiling..."
model = FANN(dim_x, dim_y, hidden_size, cell, optimizer, drop_rate, num_sents)

print "training..."


start = time.time()
g_error = 9999.9999
for i in xrange(2000):
    error = 0.0
    in_start = time.time()
    for batch_id, xy in data_x_yl_yr.items():
        X = xy[0]
        mask = xy[1]
        YL = xy[2]
        mask_y_left = xy[3]
        YR = xy[4]
        mask_y_right = xy[5]
        local_batch_size = xy[6]

        cost, sents_left, sents_right, test, test2, test3, test4, test5,test6,test7,test8,test9 = model.train(X, YL, YR, mask, mask_y_left, mask_y_right, lr, local_batch_size)
        error += cost
        
    in_time = time.time() - in_start
    # break

    #打印结果
    print "left : "
    get_data.print_sentence(sents_left, dim_y, i2w)
    print "right : "
    get_data.print_sentence(sents_right, dim_y, i2w)

    error /= len(data_x_yl_yr);
    if error < g_error:
        g_error = error

    print "Iter = " + str(i) + ", Error = " + str(error) + ", Time = " + str(in_time)
    if error <= e:
        break

print "Finished. Time = " + str(time.time() - start)

print "save model..."
save_model("FANN.model", model)
