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
lr = 1
drop_rate = 0.
batch_size = 200
read_data_batch = 200
# full_data_len = 190363

full_data_len = 200
hidden_size = [200,200]
# try: gru, lstm
cell = "gru"
# try: sgd, momentum, rmsprop, adagrad, adadelta, adam
optimizer = "adadelta" 






# batch_size = 10
# read_data_batch = 10
# # full_data_len = 190363

# full_data_len = 10

x_path = "data/200.post"
y_path = "data/200.response"
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


print "save data dic..."
save_data_dic("data/i2w200.pkl", "data/w2i200.pkl", i2w, w2i)

print "#features = ", dim_x, "#labels = ", dim_y

print "compiling..."
model = FANN(dim_x, dim_y, hidden_size, cell, optimizer, drop_rate, num_sents)
# load_error_model("GRU-200_best.model", model)

print "training..."


start = time.time()
g_error = 3.0
for i in xrange(5000):
    error = 0.0
    in_start = time.time()
    for batch_id, xy in data_x_y.items():
        X = xy[0]
        mask = xy[1]
        Y = xy[2]
        mask_y = xy[3]
        local_batch_size = xy[4]

        cost, sents, t, test, test2 = model.train(X, Y, mask, mask_y, lr, local_batch_size)
        error += cost
        # break
        
    in_b_time = time.time() - in_start
    # break

        # l,r = model.predict(data_t1[0][0], data_t1[0][1],data_t1[0][3], data_t1[0][5], 1)
        # l2,r2 = model.predict(data_t2[0][0], data_t2[0][1],data_t2[0][3], data_t2[0][5], 1)
        # l3,r3 = model.predict(data_4[0][0], data_4[0][1],data_4[0][3], data_4[0][5], 1)
        # t_sents = model.predict(test_data_x_y[0][0], test_data_x_y[0][1],test_data_x_y[0][3], batch_size)
        #打印结果

        # print "Test : "
        # get_data.print_sentence(l, dim_y, i2w)
        # get_data.print_sentence(l2, dim_y, i2w)
        # get_data.print_sentence_last_n(t_sents[0], dim_y, i2w, 5)

    error /= len(data_x_y);
    
    print "Iter = " + str(i)+ " Error = " + str(error) + ", Time = " + str(in_b_time)
    if error < g_error:
        g_error = error
        print 'new smaller cost, save param...'
        save_model("GRU_hidden200-200_post200.model", model)
    if error < 2.0:
        print "train_last :"
        get_data.print_sentence(sents, dim_y, i2w)


    if error <= e:
        break

print "Finished. Time = " + str(time.time() - start)

print "save model..."
save_model("GRU_hidden200-200_post200-final.model", model)
