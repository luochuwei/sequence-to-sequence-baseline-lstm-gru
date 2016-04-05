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
lr = 0.5
drop_rate = 0.
batch_size = 1
read_data_batch = 500
full_data_len = 190363
hidden_size = [200]
# try: gru, lstm
cell = "gru"
# try: sgd, momentum, rmsprop, adagrad, adadelta, adam
optimizer = "adadelta" 


x_path = "data/10000X.txt"
y_left_path = "data/10000r.txt"
y_right_path = "data/10000l.txt"

threshold = 0

xs, yls, yrs, i2w, w2i, tf, data_x_yl_yr = get_data.processing(x_path, y_left_path, y_right_path, threshold, 0, 10000, batch_size)
xs4, yls4, yrs4, i2w4, w2i4, tf4, data_49522 = get_data.processing(x_path, y_left_path, y_right_path, threshold, 49522, 49523, batch_size)
xs4, yls4, yrs4, i2w4, w2i4, tf4, data_49540 = get_data.processing("data/post.txt", "data/left_r.txt", "data/right_r.txt", threshold, 49540, 49540, batch_size)

# assert len(data_190353) == 1

print "#dic = " + str(len(w2i))
# print "unknown = " + str(tf["<UNknown>"])

dim_x = len(w2i)
dim_y = len(w2i)
num_sents = batch_size


print "save data dic..."
save_data_dic("data/i2w.pkl", "data/w2i.pkl", i2w, w2i)

print "#features = ", dim_x, "#labels = ", dim_y

print "compiling..."
model = FANN(dim_x, dim_y, hidden_size, cell, optimizer, drop_rate, num_sents)

print "training..."

load_model("FANN.model", model)


start = time.time()
g_error = 999999999.9999

for i in xrange(2000):
    error = 0.0
    in_start = time.time()
    in_b_start = time.time()
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
        all_error += cost

        if (batch_id+1) % 500 == 0:
            in_time = time.time() - in_start
            l,r = model.predict(data_49522[0][0], data_49522[0][1],data_49522[0][3], data_49522[0][5], 1)
            l1,r1 = model.predict(data_49540[0][0], data_49540[0][1],data_49540[0][3], data_49540[0][5], 1)
            in_b_time = time.time() - in_b_start
            in_b_start = time.time()
            # break
            #打印结果
            print "left : "
            get_data.print_sentence(l, dim_y, i2w)
            get_data.print_sentence(r1, dim_y, i2w)
            print "right : "
            get_data.print_sentence(r, dim_y, i2w)
            get_data.print_sentence(r1, dim_y, i2w)

            error /= 500.0;
            if error < g_error:
                g_error = error
                save_model("10000.model", model)

            print "Iter = " + str(i) + ", " + str(float(batch_size+1)/len(data_x_yl_yr)) + "%, Error = " + str(error) + ", Time = " + str(in_b_time)
    
    if all_error/len(data_x_yl_yr) <= e:
        break

print "Finished. Time = " + str(time.time() - start)

print "save model..."
save_model("10000.model", model)
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

#         l,r = model.predict(data_190353[0][0], data_190353[0][1],data_190353[0][3], data_190353[0][5], 1)
#         #打印结果
#         print "left : "
#         get_data.print_sentence(l, dim_y, i2w)
#         print "right : "
#         get_data.print_sentence(r, dim_y, i2w)

#         read_data_batch_error /= len(data_x_yl_yr);
#         del X_seqs
#         del yl_seqs
#         del data_x_yl_yr
#         gc.collect()
#         if read_data_batch_error < g_error:
#             g_error = read_data_batch_error
#             print 'new smaller cost, save param...'
#             save_model("FANN.model", model)

#         print "Minibatch_Iter = " + str(get_num_start)+ ", "+ str(100*(get_num_start+1)/float((full_data_len/read_data_batch)+1)) + "%, read_batch_Error = " + str(read_data_batch_error) + ", Time = " + str(in_b_time)
#     if error <= e:
#         break

# print "Finished. Time = " + str(time.time() - start)

# print "save model..."
# save_model("FANN.model", model)
