#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import os
import numpy as np

class Data:
    def __init__(self, equalizer=True, normalizer=True, skeleton=False, training_sample_size=1000, testing_sample_size=1000):
        self.equalizer = equalizer
        self.normalizer = normalizer
        self.training_sample_size = training_sample_size
        self.testing_sample_size = testing_sample_size
        self.skeleton = skeleton
        
    def equalize_light(self, image, limit=3, grid=(7,7), gray=False):
        if (len(image.shape) == 2):
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            gray = True

        clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=grid)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))

        image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        if gray: 
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return np.uint8(image)
    
    def min_max_normalization(self, input_X):
        input_X_ = (input_X - np.min(input_X))/(np.max(input_X) - np.min(input_X))
        
        return input_X_       

    def load_training_data(self):
        bone_base_path = "Dataset/"
        source = "source/"
        target = "target/"
        bone_inputs = []
        bone_outputs = []
        for img in os.listdir(bone_base_path+source)[0:self.training_sample_size]:
            im1 = cv2.bitwise_not(cv2.resize(cv2.imread(bone_base_path+source+img), (256,256)))
            im2 = cv2.bitwise_not(cv2.resize(cv2.imread(bone_base_path+target+img), (256,256)))
            if self.skeleton==False:
                bone_outputs.append(im2.reshape([256, 256,3]))
            else:
                diff = np.subtract(im1,im2)
                diff = np.where(diff<0, 0, np.where(diff>255, 255, diff))
                bone_outputs.append(diff.reshape([256, 256,3]))
            bone_inputs.append(im1.reshape([256, 256,3]))
            if self.equalizer==True:
                bone_inputs.append(self.equalize_light(im1))
                bone_outputs.append(self.equalize_light(im2))
        if self.normalizer=='min_max':
            bone_X = np.stack([self.min_max_normalization(i) for i in bone_inputs], axis=0)
            bone_Y = np.stack([self.min_max_normalization(i) for i in bone_outputs], axis=0)
        elif self.normalizer=='normal':
            bone_Y = np.stack(bone_outputs, axis=0)/255.0
            bone_X = np.stack(bone_inputs, axis=0)/255.0   
        else:
            bone_Y = np.stack(bone_outputs, axis=0)
            bone_X = np.stack(bone_inputs, axis=0)
        
        return bone_X, bone_Y

    def load_testing_data():
        bone_base_path = "Dataset/"
        source = "source/"
        target = "target/"
        bone_inputs = []
        bone_outputs = []
        for img in os.listdir(bone_base_path+source)[self.training_sample_size:self.training_sample_size+self.testing_sample_size]:
            im1 = cv2.bitwise_not(cv2.resize(cv2.imread(bone_base_path+source+img), (256,256)))
            im2 = cv2.bitwise_not(cv2.resize(cv2.imread(bone_base_path+target+img), (256,256)))
            if self.skeleton==False:
                bone_outputs.append(im2.reshape([256, 256,3]))
            else:
                bone_outputs.append((im1 - im2).reshape([256, 256,3]))
            bone_inputs.append(im1.reshape([256, 256,3]))
            if self.equalizer==True:
                bone_inputs.append(self.equalize_light(im1))
                bone_outputs.append(self.equalize_light(im2))
        bone_Y = np.stack(bone_outputs, axis=0)
        bone_X = np.stack(bone_inputs, axis=0)
        if self.normalizer==True:
            bone_X = bone_X/255.0
            bone_Y = bone_Y/255.0

        return bone_X, bone_Y