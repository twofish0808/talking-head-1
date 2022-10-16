#!/usr/bin/env python
# coding: utf-8

# In[12]:


import os
import shutil
import torch
# from torch._C import FloatTensor
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

from tha.face_morpher import FaceMorpher
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(torch.cuda.current_device()))

# In[2]:


imgFolder = 'D:/talking-head-anime-demo-master/dataSet1/img'                     #Img Path
labelFolder = 'D:/talking-head-anime-demo-master/dataSet1/label'                #Label Path
dataImgTxt = 'D:/talking-head-anime-demo-master/tryDataImg.txt'             #Output DataImg.txt
targetImgTxt = 'D:/talking-head-anime-demo-master/tryTargetImg.txt'         #Output TargetImg.txt
labelTxt = 'D:/talking-head-anime-demo-master/tryLabel.txt'                           


# In[3]:


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


class CrypkoDataset(Dataset):
    def __init__(self, imgList, labelList, resultList, transform):
        self.transform = transform
        self.imgList = imgList
        self.labelList = labelList
        self.resultList = resultList
        self.num_samples = len(self.imgList)

    def __getitem__(self,idx):



        imgList = self.imgList[idx]
        # 1. Load the image
        img = torchvision.io.read_image(imgList)
        # 2. Resize and normalize the images using torchvision.
        img = self.transform(img)
        # print("start transform")
        
        resultList = self.resultList[idx]
        target = torchvision.io.read_image(resultList)
        target = self.transform(target)
        
        label = self.labelList[idx].split(',')[:3]
        label = [float(i) for i in label]
        # print(label)
        label = torch.FloatTensor(label)
        # print(label)
        # print(label.dim())
        #label = label.unsqueeze(dim=0)
        # print(label)
        
        return img, label, target

    def __len__(self):
        return self.num_samples
    


def get_dataset(imgList, labelList, resultList):
    # 1. Resize the image to (64, 64)
    # 2. Linearly map [0, 1] to [-1, 1]
    print("get data")
    compose = [
        transforms.ToPILImage(),
        #transforms.Resize((64, 64)),
        transforms.ColorJitter(brightness=(0.9,0.9000001)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
    transform = transforms.Compose(compose)
    dataset = CrypkoDataset(imgList, labelList, resultList, transform)
    return dataset


# In[5]:


import torchvision  
print(torchvision.__version__)


# In[6]:


dataset = get_dataset(imgList, labelList, resultList)

#images = [dataset[i] for i in range(16)]
#grid_img = torchvision.utils.make_grid(images, nrow=4)
#plt.figure(figsize=(10,10))
#plt.imshow(grid_img.permute(1, 2, 0))
#plt.show()

#print(dataset[0].size())


# In[11]:



model = FaceMorpher().cuda() #第一次訓練使用這個
model.load_state_dict(torch.load('./checkpoints/face_morpher/face_morpher.pt')) #第N次訓練使用這個，保存提取

criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas = (0.9,0.999))

model.train()
n_epoch = 100000

# 準備 dataloader, model, loss criterion 和 optimizer
img_dataloader = DataLoader(dataset, batch_size=27, shuffle=True)

epoch_loss = 0
# abc=open("F:/talking-head-anime-demo-master/train_file/error.txt",mode='w')
# abc.truncate()
# abc.close()
#pose = torch.zeros(3).cuda()
#pose = pose.unsqueeze(dim=0)
last_epoch_loss=1
# 主要的訓練過程
print("start training")
while True:
    startTime = time.time()
    for epoch in range(n_epoch):
        epoch_loss = 0
        count = 1
        
        for data, label, target in img_dataloader:
            # data+=keras.layers.GaussianNoise(0.1)
            # rand=float('{:.2f}'.format(random.uniform(0.1, 0.9)))

            # compose2 = [
            # # transforms.ToPILImage(),
            # #transforms.Resize((64, 64)),
            # transforms.ColorJitter(brightness=(rand,rand+0.0001)),
            # # transforms.ToTensor(),
            # # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            # ]
            




            # if random.randrange(9)<=3:
            #     print("true")


            # num=random.uniform(0,0.0002)    
                
            img=data
            
            # img=torch.FloatTensor([img])
            # print(img)

                # unloader = transforms.ToPILImage()
                # original=img.cuda().clone()
                # original = unloader(original[0])

                # img=torchvision.transforms.Compose([transforms.ColorJitter(brightness=(0.1))])(img)
                
                # img=transforms.Compose([transforms.ToTensor()])(img2)
                # print(img.shape)
                # print(type(img))
            # else:
            #     img=data
                # img=transforms.Compose()(img)




            
            img = Variable(img).cuda()
            label = Variable(label).cuda()
            target = Variable(target).cuda()

            # cde=open("F:/talking-head-anime-demo-master/train_file/tensor.txt",mode='w')
            # print(img,file=cde)
            # cde.close()
            
            #print(img.size())
            #print(label.size())
            # print(label)
            # print(".")
            output1, alpha, color = model(img, label)
            # print(type(img))
            # print(img.dim())
            # print("label")
            # print(type(label))
            # print(label.dim())
            loss = criterion(output1, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
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

            
            if count % 50 == 0 :
                torch.save(model.state_dict(), './checkpoints/face_morpher/face_morpher.pt')
                print('save ./checkpoints/face_morpher.pt')
            if count %2000==0:
                localtime1 = time.localtime()
                result_time = time.strftime("%Y%m%d%I%M%p", localtime1)
                os.system('xcopy "D:/talking-head-anime-demo-master/checkpoints/face_morpher" "E:/facemorpher"')
                old_path="E:/facemorpher/face_morpher.pt"
                f_name="E:/facemorpher/face_morpher"+str(result_time)+".pt"
                os.rename(old_path,f_name)
            
            epoch_loss += loss.item()
            torch.cuda.empty_cache()
            print('epoch:[{}], batch:[{}/{}], loss:[{}], time:[{}:{}]'.format(epoch, count, len(img_dataloader), round(loss.item(), 4),int((time.time()-startTime)/60/60), int((time.time()-startTime)/60%60)))
            count+=1
            
        # print('epoch [{}/{}], loss:{:.5f}, time:[{}:{}]'.format(epoch+1, n_epoch, epoch_loss, int((time.time()-startTime)/60/60), int((time.time()-startTime)/60)))

        if (epoch+1) %1==0:
            localtime1 = time.localtime()
            result_time = time.strftime("%Y%m%d%I%M%p", localtime1)
            os.system('xcopy "D:/talking-head-anime-demo-master/checkpoints/face_morpher" "E:/facemorpher"')
            old_path="E:/facemorpher/face_morpher.pt"
            f_name="E:/facemorpher/face_morpher"+str(result_time)+".pt"
            os.rename(old_path,f_name)
        
        # torch.save(model.state_dict(), './checkpoints/face_morpher.pt')
            
        # print('epoch [{}/{}], loss:{:.5f}, time:[{}:{}]'.format(epoch+1, n_epoch, epoch_loss, int((time.time()-startTime)/60/60), int((time.time()-startTime)/60)))
        # print("save successfully")
    # 訓練完成後儲存 model
    torch.save(model.state_dict(), './checkpoints/face_morpher/face_morpher.pt')

    # In[ ]:





# %%
