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


_, _, i2w, w2i, tf, _ = get_data.processing(x_path, y_path, threshold, 0, 1, 1)

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
load_model("GRU-200_best.model", model)

print "training..."


start = time.time()
g_error = 999999999.9999
for i in xrange(5000):
    error = 0.0
    in_start = time.time()
    for get_num_start in xrange((full_data_len/read_data_batch)+1):
        read_data_batch_error = 0.0
        in_b_start = time.time()
        get_num_end = get_num_start*read_data_batch + read_data_batch
        if get_num_end > full_data_len:
            x = full_data_len - get_num_start*read_data_batch 
            get_num_end = get_num_start*read_data_batch + x/batch_size*batch_size
        if get_num_start*read_data_batch == full_data_len:
            break
        X_seqs, y_seqs, i2w, w2i, tf, data_x_y = get_data.processing(x_path, y_path, threshold, get_num_start*read_data_batch, get_num_end, batch_size)
        for batch_id, xy in data_x_y.items():
            X = xy[0]
            mask = xy[1]
            Y = xy[2]
            mask_y = xy[3]
            local_batch_size = xy[4]

            cost, sents = model.train(X, Y, mask, mask_y, lr, local_batch_size)
            read_data_batch_error += cost  
        
        in_b_time = time.time() - in_b_start
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

        read_data_batch_error /= len(data_x_y);
        error += read_data_batch_error
        del X_seqs
        del y_seqs
        del data_x_y
        gc.collect()
        print "Minibatch_Iter = " + str(get_num_start)+ ", "+ str(100*(get_num_start+1)/float((full_data_len/read_data_batch))) + "%, read_batch_Error = " + str(read_data_batch_error) + ", Time = " + str(in_b_time)
    if error < g_error:
        g_error = error
        print 'new smaller cost, save param...'
        save_model("GRU-200_best.model", model)        
    print i, 'iter', "cost is ", error
    # get_data.print_sentence(t_sents[0], dim_y, i2w)
    print "train_last :"
    get_data.print_sentence_last_n(sents, dim_y, i2w, 20)


    if error <= e:
        break

print "Finished. Time = " + str(time.time() - start)

print "save model..."
save_model("GRU-200_best-final.model", model)
