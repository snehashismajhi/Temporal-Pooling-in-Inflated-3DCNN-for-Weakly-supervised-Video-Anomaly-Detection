# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 17:18:27 2021

@author: Snehashis
"""
import tensorflow as tf

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
