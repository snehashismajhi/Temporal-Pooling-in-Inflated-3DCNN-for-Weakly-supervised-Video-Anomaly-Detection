# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 20:10:44 2020

@author: Snehashis#
"""

import os
from keras import backend as K
print(K.backend())
from keras.optimizers import Adam
import tensorflow as tf

from sklearn.metrics import roc_auc_score, roc_curve,auc
#%%
import os
import sys
import numpy as np
from keras.callbacks import Callback
import multiprocessing
from DL_TRAIN import *
from DL_TEST import *
from keras.utils import plot_model
from model import *
#from sklearn.metrics import confusion_matrix
#%%

seed = 7
np.random.seed(seed)

pool_type = 'mean'
class Metrics(Callback):

    def on_train_begin(self, logs={}):
        self.ALL_AUC = []
        self.ALL_EPOCH = []
        self.ALL_acc1 = []
        self.ALL_acc2 = []
        self.ALL_acc3 = []
        self.ALL_acc4 = []
        self.ALL_Avg_ACC = []
        
    def get_GT(self,test_file):
        test_file=test_file.split()
        video_name=test_file[0][:-4]
        # class_folder = test_file[1]
        s1 = int(test_file[2])
        e1 = int(test_file[3])
        s2 = int(test_file[4])
        e2 = int(test_file[5])
        Test_video_name = np.load("E:\Research\code\Detection\Test_video_name.npy")
        Test_frame_number = np.load("E:\Research\code\Detection\Test_frame_number.npy")
        video_index = int(np.argwhere(Test_video_name==video_name))
        gt = np.zeros((Test_frame_number[video_index], 1))  # Initially all normal     #NORMAL = 1  # ABNORMAL =0

        if s1 != -1 and e1 != -1:
            gt[s1:e1, 0] = 1
        if s2 != -1 and e2 != -1:
            gt[s2:e2, 0] = 1
        return gt

 
    def my_metrics_DETECTION(self):
        test_gen= DataLoader_test_detect(pool_type)
        score=model.predict_generator(test_gen)

        ############ DETECTION PART ###########

        detection_score=np.asarray(score)
        temp_annotation = 'E:\Research\code\Detection\Temporal_Anomaly_Annotation_for_Testing_Videos.txt'
        test_files = [i.strip() for i in open(temp_annotation).readlines()]
        ALL_GT = np.array([])
        ALL_score = np.array([])
        for i in range(len(test_files)):
            video_file = test_files[i]
            video_file = video_file.split()
            video_name = video_file[0][:-4]
            # print(video_name)

            video_segment_score = detection_score[i]
            video_GT = self.get_GT(test_files[i])
            video_score = np.array([])
            for k in range(32):
                dummy_score = np.repeat(video_segment_score[k, 0], np.floor(video_GT.shape[0] / 32))
                video_score = np.concatenate([video_score, dummy_score])

            if video_GT.shape[0] % 32 != 0:
                dummy_remain_score = np.repeat(video_segment_score[31, 0],
                                               video_GT.shape[0] - np.floor(video_GT.shape[0] / 32) * 32)
                video_score = np.concatenate([video_score, dummy_remain_score])

            video_GT = np.squeeze(video_GT, axis=1)
            ALL_GT = np.concatenate([ALL_GT, video_GT])
            ALL_score = np.concatenate([ALL_score, video_score])

        AUC = roc_auc_score(ALL_GT, ALL_score)
        return AUC





    def on_epoch_end(self, epoch, logs={}):



        if (epoch + 1) % 50== 0:

            AUC = self.my_metrics_DETECTION()


            print("###############  AUC : " + str(AUC))

            self.ALL_EPOCH.append(epoch)
            self.ALL_AUC.append(AUC)

            AUC_new = np.asarray(self.ALL_AUC)
            EPOCH_new = np.asarray(self.ALL_EPOCH)
            # acc_avg_new = np.asarray(self.ALL_Avg_ACC)
            total = np.asarray(list(zip(EPOCH_new, AUC_new)))
            np.savetxt("E:\\Research\\code\\ICCCNT\\UCF\\AUC\\AUC_"+pool_type+".txt", total, delimiter=',')
            

            prev_auc = np.max(np.array(AUC_new))
            if AUC >= prev_auc:
                model.save('E:\\Research\\code\\ICCCNT\\UCF\\model\\MIL_model_weights_'+pool_type+'.h5')
                print('MIL model weights saved Sucessfully')




        return
    
    
    

def custom_objective(y_true, y_pred):
    'Custom Objective function'
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    
    # y_true = tf.keras.layers.Flatten(y_true)
    # y_pred = tf.keras.layers.Flatten(y_pred)
    
    n_seg = 32  # Because we have 32 segments per video.
    nvid = 60
    n_exp = nvid / 2
    Num_d=32*nvid


    sub_max = tf.ones_like(y_pred) # sub_max represents the highest scoring instants in bags (videos).
    sub_sum_labels = tf.ones_like(y_true) # It is used to sum the labels in order to distinguish between normal and abnormal videos.
    sub_sum_l1=tf.ones_like(y_true)  # For holding the concatenation of summation of scores in the bag.
    sub_l2 = tf.ones_like(y_true) # For holding the concatenation of L2 of score in the bag.

    for ii in range(0, nvid, 1):
        # For Labels
        mm = y_true[ii * n_seg : ii * n_seg + n_seg]
        sub_sum_labels = tf.concat([sub_sum_labels, [tf.reduce_sum(mm)]],0)# Just to keep track of abnormal and normal vidoes

        # For Features scores
        Feat_Score = y_pred[ii * n_seg : ii * n_seg + n_seg]
        sub_max = tf.concat([sub_max, [tf.reduce_max(Feat_Score, 0)]], 0)     # Keep the maximum score of scores of all instances in a Bag (video)
        sub_sum_l1 = tf.concat([sub_sum_l1,[tf.reduce_sum(Feat_Score)]], 0)   # Keep the sum of scores of all instances in a Bag (video)
        
        z1 = tf.ones_like(Feat_Score)
        z2 = tf.concat([z1, Feat_Score], 0)
        z3 = tf.concat([Feat_Score, z1], 0)
        z_22 = z2[31:]
        z_44 = z3[:33]
        z = z_22 - z_44
        z = z[1:32]
        z = tf.reduce_sum(tf.math.square(z))
        sub_l2 = tf.concat([sub_l2, [z]], 0)
        
        


    # sub_max[Num_d:] means include all elements after Num_d.
    # AllLabels =[2 , 4, 3 ,9 ,6 ,12,7 ,18 ,9 ,14]
    # z=x[4:]
    #[  6.  12.   7.  18.   9.  14.]

    sub_score = sub_max[Num_d:]  # We need this step since we have used T.ones_like
    F_labels = sub_sum_labels[Num_d:] # We need this step since we have used T.ones_like
    #  F_labels contains integer 32 for normal video and 0 for abnormal videos. This because of labeling done at the end of "load_dataset_Train_batch"



    # AllLabels =[2 , 4, 3 ,9 ,6 ,12,7 ,18 ,9 ,14]
    # z=x[:4]
    # [ 2 4 3 9]... This shows 0 to 3 elements

    sub_sum_l1 = sub_sum_l1[Num_d:] # We need this step since we have used T.ones_like
    sub_sum_l1 = sub_sum_l1[:int(n_exp)]
    sub_l2 = sub_l2[Num_d:]         # We need this step since we have used T.ones_like
    sub_l2 = sub_l2[:int(n_exp)]
    zero = tf.constant(0, dtype=tf.float32)
    
    # indx_nor = K.cast(K.not_equal(tf.where(tf.equal(F_labels, 32)), 0), "int64")
    # indx_abn =  K.cast(K.not_equal(tf.where(tf.equal(F_labels, 0)), 0), "int64") # checck for zero or 1
    
    indx_nor = tf.where(tf.equal(F_labels, 32))
    indx_abn = tf.where(tf.equal(F_labels, 0)) 
    
    n_Nor=n_exp
    
    # Sub_Nor = sub_score[indx_nor] # Maximum Score for each of abnormal video
    # Sub_Abn = sub_score[indx_abn] # Maximum Score for each of normal video
    
    Sub_Nor = tf.gather_nd(sub_score, indx_nor)
    Sub_Abn = tf.gather_nd(sub_score, indx_abn)
    z = tf.ones_like(y_true)
    for ii in range(0, int(n_Nor), 1):
        # sub_z = tf.reduce_max(1 - Sub_Abn + Sub_Nor[ii], 0)
        sub_z = tf.math.maximum(1 - Sub_Abn + Sub_Nor[ii], 0)
        z = tf.concat([z, [tf.reduce_sum(sub_z)]], 0)


    z = z[Num_d:]  # We need this step since we have used T.ones_like
    z = tf.reduce_mean(z) + 0.00008 * tf.reduce_sum(sub_sum_l1) + 0.00008 * tf.reduce_sum(sub_l2)  # Final Loss f

    return z







lrn = 0.0001
l3 = 0.001
nuron = 96
# adagrad=Adagrad(lr=lrn, epsilon=1e-08)
adam=Adam(lr=lrn,beta_1=0.9, beta_2=0.999,decay=0.0)
print(pool_type)
model=MIL_model(l3,nuron)
model.summary()
model.compile(loss=custom_objective,optimizer=adam)

metrics = Metrics()
print("Starting training...")
model.summary()
# num_epoch=20000

# ########################################## TRAINING #####################################################
# stp_epc = 1  # int(1610/60)
# train_generator = DataLoader_MIL_train(pool_type)
# loss = model.fit_generator(train_generator, steps_per_epoch=stp_epc, epochs=num_epoch, verbose=1,use_multiprocessing=False,workers=3,callbacks=[metrics])#,lr_scheduler])

