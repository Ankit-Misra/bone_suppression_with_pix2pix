#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.measure.entropy import shannon_entropy
from skimage.filters.rank import entropy
from skimage.morphology import disk
from math import log10, sqrt 
import cv2
import imquality.brisque as brisque

# In[ ]:


def PSNR(actuals, predicted):
    mse = np.mean((actuals - predicted) ** 2) 
    if(mse == 0):
        return 100
    max_pixel = 1.0
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr

def calculate_psnr(actuals, predicted):
    psnr = []
    for i in range(0,predicted.shape[0]):
        psnr.append(PSNR(actuals[i], predicted[i]))
    #avg_psnr = round(psnr/predicted.shape[0],4)
    #print("Average PSNR = "+str(psnr))
    return psnr

def calculate_ssim(actuals, predicted):
    ssim_noise = []
    for i in range(0,predicted.shape[0]):
        ssim_noise.append(ssim(actuals[i], predicted[i], data_range=predicted[i].max() - predicted[i].min(), multichannel=True))
    #avg_ssim = round(ssim_noise/predicted.shape[0],4)
    #print("Average SSIM = "+str(ssim_noise))
    return ssim_noise
    
def calculate_entropy(actuals, predicted):
    entropy_org = []
    entropy_gen = []
    for i in range(0,predicted.shape[0]):
        entropy_org.append(shannon_entropy(actuals[i]))
        entropy_gen.append(shannon_entropy(predicted[i]))
    #avg_entropy_org = round(entropy_org/predicted.shape[0],4)
    #avg_entropy_gen = round(entropy_gen/predicted.shape[0],4)
    #print("Avg original entropy = "+str(avg_entropy_org)+" Avg generated entropy = "+str(avg_entropy_gen))
    return entropy_org,entropy_gen

def calculate_variance(actuals, predicted):
    lap_org = []
    lap_gen = []
    for i in range(0,predicted.shape[0]):
        lap_org.append(cv2.Laplacian(actuals[i], cv2.CV_64F).var())
        lap_gen.append(cv2.Laplacian(predicted[i], cv2.CV_32F).var())
    #avg_lap_org = round(lap_org/predicted.shape[0],4)
    #avg_lap_gen = round(lap_gen/predicted.shape[0],4)
    #print("Avg original variance = "+str(avg_lap_org)+" Avg generated variance = "+str(avg_lap_gen))
    return lap_org, lap_gen

def mse(imageA,imageB):
    err = np.sum((imageA.astype('float')-imageB.astype('float'))**2)
    err = err/float(imageA.shape[0]*imageA.shape[1])
    return err

def calculate_mse(actuals, predicted):
    sum_mse=[]
    for i in range(0,predicted.shape[0]):
        sum_mse.append(mse(actuals[i], predicted[i]))
    #avg_mse = round(sum_mse/predicted.shape[0],4)
    #print("Average MSE = "+str(avg_mse))
    return sum_mse

def calculate_nmse(actuals, predicted):
    avg_ratio = []
    for i in range(0, predicted.shape[0]):
        avg_ratio.append(np.sum(np.square(actuals[i] - predicted[i]))/np.sum(np.square(actuals[i])))
    #avg_ratio = round(avg_ratio/predicted.shape[0],4)
    #print("Average BSR = "+str(avg_ratio))
    return avg_ratio

def calculate_brisque(predicted):
    brisque_score = []
    for i in predicted:
        brisque_score.append(brisque.score(i))
    return brisque_score