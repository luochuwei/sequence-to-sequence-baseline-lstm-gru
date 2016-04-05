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
lr = 2.0
drop_rate = 0.
batch_size = 500
read_data_batch = 8000
# full_data_len = 190363

full_data_len =10000
hidden_size = [100,200]
# try: gru, lstm
cell = "gru"
# try: sgd, momentum, rmsprop, adagrad, adadelta, adam
optimizer = "adadelta" 

x_path = "data/SMT-train-8000.post"
y_path = "data/SMT-train-8000.response"

threshold = 0

xs, ys, i2w, w2i, tf, data_x_y = get_data.processing(x_path, y_path, threshold)
# txs, tys, data_tx_ty = get_data.test_processing

# print "#dic = " + str(len(w2i))
# # print "unknown = " + str(tf["<UNknown>"])

# dim_x = len(w2i)
# dim_y = len(w2i)
# num_sents = batch_size


# print "save data dic..."
# save_data_dic("data/i2w10000.pkl", "data/w2i10000.pkl", i2w, w2i)

# print "#features = ", dim_x, "#labels = ", dim_y

# print "compiling..."
# model = FANN(dim_x, dim_y, hidden_size, cell, optimizer, drop_rate, num_sents)
# # load_model("FANN1000.model", model)

# print "training..."


# start = time.time()
# g_error = 999999999.9999
# for i in xrange(2000):
#     error = 0.0
#     in_start = time.time()
#     for get_num_start in xrange((full_data_len/read_data_batch)+1):
#         read_data_batch_error = 0.0
#         in_b_start = time.time()
#         get_num_end = get_num_start*read_data_batch + read_data_batch
#         if get_num_end > full_data_len:
#             x = full_data_len - get_num_start*read_data_batch 
#             get_num_end = get_num_start*read_data_batch + x/batch_size*batch_size
#         if get_num_start*read_data_batch == full_data_len:
#             break
#         X_seqs, yl_seqs, yr_seqs, i2w, w2i, tf, data_x_yl_yr = get_data.processing(x_path, y_left_path, y_right_path, threshold, get_num_start*read_data_batch, get_num_end, batch_size)
#         for batch_id, xy in data_x_yl_yr.items():
#             X = xy[0]
#             mask = xy[1]
#             YL = xy[2]
#             mask_y_left = xy[3]
#             YR = xy[4]
#             mask_y_right = xy[5]
#             local_batch_size = xy[6]

#             cost, sents_left, sents_right, test, test2, test3, test4, test5,test6,test7,test8,test9 = model.train(X, YL, YR, mask, mask_y_left, mask_y_right, lr, local_batch_size)
#             read_data_batch_error += cost
#             error += cost   
        
#         in_b_time = time.time() - in_b_start
#         # break

#         # l,r = model.predict(data_t1[0][0], data_t1[0][1],data_t1[0][3], data_t1[0][5], 1)
#         # l2,r2 = model.predict(data_t2[0][0], data_t2[0][1],data_t2[0][3], data_t2[0][5], 1)
#         # l3,r3 = model.predict(data_4[0][0], data_4[0][1],data_4[0][3], data_4[0][5], 1)
#         l3,r3 = model.predict(data_4[0][0], data_4[0][1],data_4[0][3], data_4[0][5], 500)
#         #打印结果
#         print "left : "
#         # get_data.print_sentence(l, dim_y, i2w)
#         # get_data.print_sentence(l2, dim_y, i2w)
#         get_data.print_sentence_last_100(l3, dim_y, i2w)
#         print "right : "
#         # get_data.print_sentence(r, dim_y, i2w)
#         # get_data.print_sentence(r2, dim_y, i2w)
#         get_data.print_sentence_last_100(r3, dim_y, i2w)


#         read_data_batch_error /= len(data_x_yl_yr);
#         del X_seqs
#         del yl_seqs
#         del data_x_yl_yr
#         gc.collect()
#         if read_data_batch_error < g_error:
#             g_error = read_data_batch_error
#             print 'new smaller cost, save param...'
#             save_model("FANN10000_batch1000_best.model", model)

#         print "Minibatch_Iter = " + str(get_num_start)+ ", "+ str(100*(get_num_start+1)/float((full_data_len/read_data_batch)+1)) + "%, read_batch_Error = " + str(read_data_batch_error) + ", Time = " + str(in_b_time)
#     if error <= e:
#         break

# print "Finished. Time = " + str(time.time() - start)

# print "save model..."
# save_model("FANN10000_batch1000_final.model", model)
