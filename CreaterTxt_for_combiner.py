#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import shutil
import torch
from tha.face_morpher import FaceMorpher
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
print(torch.cuda.is_available())


# In[2]:


imgFolder = 'D:/talking-head-anime-demo-master/dataSet1/img'                     #Img Path
labelFolder = 'D:/talking-head-anime-demo-master/dataSet1/label'                #Label Path
dataImgTxt = 'D:/talking-head-anime-demo-master/tryDataImg3.txt'             #Output DataImg.txt
targetImgTxt = 'D:/talking-head-anime-demo-master/tryTargetImg3.txt'         #Output TargetImg.txt
labelTxt = 'D:/talking-head-anime-demo-master/tryLabel3.txt'               


# In[3]:


imgList = []
resultList = []

def find_dir(path):
    for fd in os.listdir(path):
        full_path = os.path.join(path, fd)
        if fd.split('_')[-1] == '0.png':
            for _ in range(141):
                imgList.append(full_path)
        elif fd.split('.')[-1] == 'png' and int(fd.split('_')[-1].split('.')[0])<=141 :
            print(full_path)
            resultList.append(full_path)
        
        if os.path.isdir(full_path):
            find_dir(full_path)
                     
find_dir(imgFolder)


# In[4]:


labelList = []

def find_dir(path):
    for fd in os.listdir(path):
        full_path = os.path.join(path, fd)
        if fd.split('.')[-1] == 'txt' and int(fd.split('_')[-1].split('.')[0])<=141 and int(fd.split('_')[-1].split('.')[0])>0:
            a = []
            with open(full_path, 'r') as f:
                a = f.readlines()
            print(a[0][:-1])
            labelList.append(a[0][:-1])
        if os.path.isdir(full_path):
            find_dir(full_path)
                     
find_dir(labelFolder)


# In[5]:


with open(dataImgTxt, 'w', encoding="utf-8") as f:
    for i in imgList:
        f.write(i+"\n")
        
with open(targetImgTxt, 'w', encoding="utf-8") as f:
    for i in resultList:
        f.write(i+"\n")

with open(labelTxt, 'w', encoding="utf-8") as f:
    for i in labelList:
        f.write(i+"\n")


# In[ ]:




