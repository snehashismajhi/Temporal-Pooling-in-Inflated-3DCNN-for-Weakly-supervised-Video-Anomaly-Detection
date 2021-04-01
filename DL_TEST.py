import numpy as np
import os
from keras.utils import Sequence,to_categorical
from keras.preprocessing import image
import glob
import h5py
from sklearn.preprocessing import normalize




class DataLoader_test_detect(Sequence):

    def __init__(self,pool_type):
        self.batch_size = 10
        self.temp_annotation =  'E:\Research\code\Detection\Temporal_Anomaly_Annotation_for_Testing_Videos.txt'
        self.test_files = [i.strip() for i in open(self.temp_annotation).readlines()]
        self.path_frame_feature= "E:\\Research\\Features\\UCF\\I3D_center_crop_GAP\\"
        self.pool_type = pool_type
    def __len__(self):
        return int(len(self.test_files)/self.batch_size)

    def _name_to_int(self,class_name):
       if class_name == 'Abuse':
           integer= 0
       elif class_name == 'Arrest':
           integer= 1
       elif class_name == 'Arson':
           integer= 2
       elif class_name == 'Assault':
           integer= 3
       elif class_name == 'Burglary':
           integer= 4
       elif class_name == 'Explosion':
           integer= 5
       elif class_name == 'Fighting':
           integer= 6
       elif class_name == 'Normal_Videos_event' or class_name == 'Training_Normal_Videos_Anomaly' or class_name == 'Testing_Normal_Videos_Anomaly':
           integer= 7
       elif class_name == 'RoadAccidents':
           integer= 8
       elif class_name == 'Robbery':
           integer= 9
       elif class_name == 'Shooting':
           integer= 10
       elif class_name == 'Shoplifting':
           integer= 11
       elif class_name == 'Stealing':
           integer= 12
       elif class_name == 'Vandalism':
           integer= 13
       return integer

    def _get_GT_recognition(self,test_file):
        test_file=test_file.split()
        video_name=test_file[0][:-4]
        class_folder = test_file[1]
        if class_folder == 'Normal':
            class_folder = 'Testing_Normal_Videos_Anomaly'
        label= self._name_to_int(class_folder)

        return label


    def _get_GT(self,test_file):
        test_file=test_file.split()
        video_name=test_file[0][:-4]
        # class_folder = test_file[1]
        s1 = int(test_file[2])
        e1 = int(test_file[3])
        s2 = int(test_file[4])
        e2 = int(test_file[5])
        # if class_folder == 'Normal':
        #     class_folder = 'Testing_Normal_Videos_Anomaly'
        # video_path = '/data/stars/user/smajhi/AAR_task/videos/'
        # video_path = video_path + class_folder + '/' + video_name + '/'
        # images = sorted(os.listdir(video_path))
        Test_video_name = np.load("E:\Research\code\Detection\Test_video_name.npy")
        Test_frame_number = np.load("E:\Research\code\Detection\Test_frame_number.npy")
        video_index = int(np.argwhere(Test_video_name==video_name))
        gt = np.zeros((Test_frame_number[video_index], 1))  # Initially all normal     #NORMAL = 1  # ABNORMAL =0

        if s1 != -1 and e1 != -1:
            gt[s1:e1, 0] = 1
        if s2 != -1 and e2 != -1:
            gt[s2:e2, 0] = 1
        return gt



    def __getitem__(self, item):
        batch_video=self.test_files[item * self.batch_size : (item+1) * self.batch_size]
        feature = [self._get_feature(i) for i in batch_video]
        GT=[self._get_GT(i) for i in batch_video]


        feature=np.asarray(feature)


        GT=np.asarray(GT)

        return feature,GT




    def _get_feature(self,test_file):
        # print(video_name)
        test_file=test_file.split()
        video_name=test_file[0][:-4]
        # print(video_name)
        class_name = test_file[1]
        if class_name == 'Normal':
            class_name = 'Testing_Normal_Videos_Anomaly'
        f = h5py.File(self.path_frame_feature + class_name + "\\" + video_name + '.h5', 'r')
        feature = np.array(f['features'])
        feature=np.squeeze(feature,axis=0)
        feature=np.squeeze(feature,axis=3)
        feature=np.squeeze(feature,axis=2)
        # feature=np.reshape(feature,(feature.shape[0],feature.shape[1]*feature.shape[2]))
        if self.pool_type == 'max':
            feature = np.max(feature, axis=1)
        elif self.pool_type == 'min':
            feature = np.min(feature, axis=1)
        elif self.pool_type == 'minmax':
            feature_max = np.max(feature, axis=1)
            feature_min = np.min(feature, axis=1)
            feature = np.concatenate([feature_max,feature_min],axis=1)
        elif self.pool_type == 'mean':
            feature = np.mean(feature, axis=1)
        feature_norm=feature#normalize(feature,norm='l2')
        segment_size=32

        if feature_norm.shape[0] < segment_size:
            feature_segment = feature_norm
            i = feature_norm.shape[0]
            k = feature_norm.shape[0] - 1
            while i < segment_size:
                # print(i)
                feature_segment = np.vstack([feature_segment, feature_norm[k]])
                i=i+1
        else:
            feature_segment=[]
            for i in range(segment_size):
                j=int(feature_norm.shape[0]/segment_size)
                temp=feature_norm[i*j:(i+1)*j]
                feature_segment.append(np.max(temp,axis=0))
            feature_segment=np.asarray(feature_segment)



        return feature_segment
