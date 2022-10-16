#!/usr/bin/env python
# coding: utf-8

# In[12]:


import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
# from tensorflow import keras
import time
import numpy as np
from matplotlib.pyplot import imshow
import random

from torchvision.transforms.transforms import ToTensor
from tha.perceptual import perceptual_loss,getloss
from tha.face_morpher import FaceMorpher
from tha.two_algo_face_rotator import TwoAlgoFaceRotator

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(torch.cuda.current_device()))

# In[2]:


imgFolder = 'D:/talking-head-anime-demo-master/dataSet1/img'                     #Img Path
labelFolder = 'D:/talking-head-anime-demo-master/dataSet1/label'                #Label Path
dataImgTxt = 'D:/talking-head-anime-demo-master/tryDataImg2.txt'             #Output DataImg.txt
targetImgTxt = 'D:/talking-head-anime-demo-master/tryTargetImg2.txt'         #Output TargetImg.txt
labelTxt = 'D:/talking-head-anime-demo-master/tryLabel2.txt'          


# In[3]:

#讀取CreateTxt.py建立之TXT
with open(dataImgTxt, 'r') as f:
    imgList = f.readlines()

for i in range(len(imgList)):
    imgList[i] = imgList[i][:-1]    

with open(targetImgTxt, 'r') as f:
    resultList = f.readlines()

for i in range(len(resultList)):
    resultList[i] = resultList[i][:-1]

with open(labelTxt, 'r') as f:
    labelList = f.readlines()

for i in range(len(labelList)):
    labelList[i] = labelList[i][:-1]    

#print(imgList[-1])    
#print(resultList[-1])    
#print(labelList[-1])


# In[4]:

#呼叫資料處理函數，進行資料整理，並轉為Tensor，label部分僅讀取需要的三個數字(詳見下方)
class CrypkoDataset(Dataset):
    def __init__(self, imgList, labelList, resultList, transform):
        self.transform = transform
        self.imgList = imgList
        self.labelList = labelList
        self.resultList = resultList
        self.num_samples = len(self.imgList)

    def __getitem__(self,idx):



        imgList = self.imgList[idx]
        img = torchvision.io.read_image(imgList)
        img = self.transform(img)
        
        resultList = self.resultList[idx]
        target = torchvision.io.read_image(resultList)
        target = self.transform(target)
        
        #label 為face_morpher使用之label，讀取labelList(範例:0.358320,0.104861,0.555508,-11,-12,3)的前三個數字，也就是[0.358320,0.104861,0.555508]
        #並把他建立為float格式的Tensor
        #但在此程式僅作為第一階段(使用face_morpher輸出影像)之後用該照片來尋練face_rotator
        label = self.labelList[idx].split(',')[:3]
        label = [float(i) for i in label]
        label = torch.FloatTensor(label)

        #label2 為face_rotaror使用之label，讀取labelList(範例:0.358320,0.104861,0.555508,-11,-12,3)的前三個數字，也就是[-11,-12,3]
        #並把他建立為int格式的Tensor        
        label2=self.labelList[idx].split(',')[3:]
        # label2=self.labelList[idx].split(',')
        # print(label2)
        label2 = [float(i) for i in label2]
        label2= torch.FloatTensor(label2)
        # print(label2)
        label2=torch.div(label2,15)
        # print(label2)

        #回傳處理後的東西
        return img, label, target ,label2

    def __len__(self):
        return self.num_samples
    

#此為資料處理函數，經調整亮度後，將圖片轉為Tensor(transform.Compose(compose))
def get_dataset(imgList, labelList, resultList):
    compose = [
        transforms.ToPILImage(),
        #亮度調整
        transforms.ColorJitter(brightness=(0.90,0.9000001)),
        transforms.ToTensor(),
    ]
    transform = transforms.Compose(compose)
    dataset = CrypkoDataset(imgList, labelList, resultList, transform)
    return dataset


# In[5]:


import torchvision  
print(torchvision.__version__)


# In[6]:

#資料處理，將處理過的東西存入dataset
dataset = get_dataset(imgList, labelList, resultList)



# In[11]:


#讀取和建立FaceMorpher模型
model = FaceMorpher().cuda() 
model.load_state_dict(torch.load('./checkpoints/face_morpher/face_morpher.pt')) 

#讀取和建立TwoAlgoFaceRotator模型
model2= TwoAlgoFaceRotator().cuda()
model2.load_state_dict(torch.load('./checkpoints/two_algo_face_rotator/two_algo_face_rotator.pt'))

#定義loss
criterion = nn.L1Loss()
# criterion = perceptual_loss()
# criterion1=nn.MSELoss()
#Adam為一種梯度下降優化演算法
optimizer = torch.optim.Adam(model2.parameters(), lr=0.00015, betas = (0.5,0.999))

model.eval()
model2.train()
n_epoch = 100000

# 準備 dataloader, model, loss criterion 和 optimizer
img_dataloader = DataLoader(dataset, batch_size=7, shuffle=True)

epoch_loss = 0
last_epoch_loss=1
# 主要的訓練過程
print("start training")
while True:
    startTime = time.time()
    for epoch in range(n_epoch):
        epoch_loss = 0
        count = 1
        
        for data, label, target, label2 in img_dataloader:


            #隨機加入雜訊
            num=random.uniform(0,0.000015)        
            img = data + num*torch.randn(256,256)


            #+是tensor的外包装，也就像錢和錢包那種概念，裡面還有其他東西
            img = Variable(img).cuda()
            label = Variable(label).cuda()
            label2=Variable(label2).cuda()
            target = Variable(target).cuda()


            #img經過face_morpher處理輸出為output1，其他兩個變數為演算法衍伸物，詳細return內容可見tha/face_morpher.py
            output1, alpha, color = model(img, label)
            
            torch.cuda.empty_cache()
            #output1經過face_rotator處理輸出為color_changed和resampled，其他兩個變數為演算法衍伸物，詳細return內容和演算法可見tha/two_algo_face_rotator.py
            color_changed, resampled, color_change, alpha_mask, grid_change, grid=model2(output1,label2)
            torch.cuda.empty_cache()

            #將兩個演算法的loss算出來
            loss1 = getloss(color_changed, target)
            loss1=loss1+criterion(color_changed,target)
            # loss1=loss1+criterion1(color_changed,target)
            loss2=getloss(resampled,target)
            loss2=loss2+criterion(resampled,target)
            # loss2=loss2+criterion1(resampled,target)

            #之後取平方的平均開根號後回傳給訓練程式
            # loss=(((loss2**2)+(loss1**2))**0.5)/2
            loss=(loss1+loss2)/2
            # loss=loss2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #輸出所有演算法衍伸物和結果，以方便觀察(可寫可不寫)
            unloader = transforms.ToPILImage()
            original=img.cpu().clone()
            original = unloader(original[0]).save("./train_file/original.png")
            color1 = color.cpu().clone()
            color1 = unloader(color1[0]).save("./train_file/color.png")
            alpha1 = alpha.cpu().clone()
            alpha1 = unloader(alpha1[0]).save("./train_file/alpha.png")
            output_image = output1.cpu().clone()
            output_image = unloader(output_image[0]).save("./train_file/output_image.png")
            target1 = target.cpu().clone()
            target1 = unloader(target1[0]).save("./train_file/target1.png")
            color_cd1 = color_changed.cpu().clone()
            color_cd1 = unloader(color_cd1[0]).save("./train_file/color_changed.png")
            resampled1 = resampled.cpu().clone()
            resampled1 = unloader(resampled1[0]).save("./train_file/resampled.png")
            color_change1 = color_change.cpu().clone()
            color_change1 = unloader(color_change1[0]).save("./train_file/color_change.png")
            alpha_mask1 = alpha_mask.cpu().clone()
            alpha_mask1 = unloader(alpha_mask1[0]).save("./train_file/alpha_mask.png")

            #計算epoch的loss之和，並準備下一次訓練
            epoch_loss += loss.item()
            torch.cuda.empty_cache()

            print('epoch:[{}], batch:[{}/{}], loss:[{}], lossA[{}], lossB[{}] ,time:[{}:{}]'.format(epoch, count, len(img_dataloader), loss.item(),loss1,loss2,int((time.time()-startTime)/60/60), int((time.time()-startTime)/60%60)))
            count+=1

            #每執行50次進行一次存檔
            if count % 50 == 0 :
                torch.save(model2.state_dict(), './checkpoints/two_algo_face_rotator/two_algo_face_rotator.pt')
                print('save successfully')
            
            if count % 1500 ==0:
                localtime1 = time.localtime()
                result_time = time.strftime("%Y%m%d%I%M%p", localtime1)
                os.system('xcopy "D:/talking-head-anime-demo-master/checkpoints/two_algo_face_rotator" "E:/face_rotater"')
                old_path="E:/face_rotater/two_algo_face_rotator.pt"
                f_name="E:/face_rotater/two_algo_face_rotator_more"+str(result_time)+".pt"
                os.rename(old_path,f_name)
            
        #完成一個epoch後進行存檔        
        torch.save(model2.state_dict(),"./checkpoints/two_algo_face_rotator/two_algo_face_rotator.pt")
        print("save_successfully")


        #每500個epoch自動備份(若不需要可註解掉)
        # if (epoch+1) %8==0:
        #         localtime1 = time.localtime()
        #         result_time = time.strftime("%Y%m%d%I%M%p", localtime1)
        #         os.system('xcopy "E:/talking-head-anime-demo-master/checkpoints/two_algo_face_rotator" "E:/face_rotater"')
        #         old_path="E:/face_rotater/two_algo_face_rotator.pt"
        #         f_name="E:/face_rotater/two_algo_face_rotator_more"+str(result_time)+".pt"
        #         os.rename(old_path,f_name)
        if (epoch+1) %20==0:
                localtime1 = time.localtime()
                result_time = time.strftime("%Y%m%d%I%M%p", localtime1)
                os.system('xcopy "D:/talking-head-anime-demo-master/checkpoints/two_algo_face_rotator" "E:/face_rotater"')
                old_path="E:/face_rotater/two_algo_face_rotator.pt"
                f_name="E:/face_rotater/two_algo_face_rotator_more"+str(result_time)+".pt"
                os.rename(old_path,f_name)
        #輸出本輪解果
        print('epoch [{}/{}], loss:{:.5f}, time:[{}:{}]'.format(epoch+1, n_epoch, epoch_loss, int((time.time()-startTime)/60/60), int((time.time()-startTime)/60)))
    torch.save(model2.state_dict(),"./checkpoints/two_algo_face_rotator/two_algo_face_rotator.pt")

    # In[ ]:




