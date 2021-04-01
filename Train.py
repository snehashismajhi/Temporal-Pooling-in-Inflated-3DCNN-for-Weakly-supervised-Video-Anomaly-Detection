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
from keras.utils import plot_model
import multiprocessing
from DlTrain import *
from DlTest import *
from Model import *
from Objective import *

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
    
    
    

lrn = 0.0001
l3 = 0.001
nuron = 96

adam=Adam(lr=lrn,beta_1=0.9, beta_2=0.999,decay=0.0)
print(pool_type)
model=MIL_model(l3,nuron)
model.summary()
model.compile(loss=custom_objective,optimizer=adam)

metrics = Metrics()
print("Starting training...")
model.summary()
num_epoch=20000

# ########################################## TRAINING #####################################################
stp_epc = 1  # int(1610/60)
train_generator = DataLoader_MIL_train(pool_type)
model.fit_generator(train_generator, steps_per_epoch=stp_epc, epochs=num_epoch, verbose=1,use_multiprocessing=False,workers=3,callbacks=[metrics])#,lr_scheduler])

