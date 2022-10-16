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
import time
import numpy as np
from matplotlib.pyplot import imshow
import random

from torchvision.transforms.transforms import ToTensor

from tha.face_morpher import FaceMorpher
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(torch.cuda.current_device()))
print(torchvision.__version__)


imgFolder = 'D:/talking-head-anime-demo-master/dataSet1/img'                     #Img Path
labelFolder = 'D:/talking-head-anime-demo-master/dataSet1/label'                #Label Path
dataImgTxt = 'D:/talking-head-anime-demo-master/tryDataImg.txt'             #Output DataImg.txt
targetImgTxt = 'D:/talking-head-anime-demo-master/tryTargetImg.txt'         #Output TargetImg.txt
labelTxt = 'D:/talking-head-anime-demo-master/tryLabel.txt'                           


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
        resultList = self.resultList[idx]
        target = torchvision.io.read_image(resultList)
        target = self.transform(target)
        label = self.labelList[idx].split(',')[:3]
        label = [float(i) for i in label]
        label = torch.FloatTensor(label)
        
        return img, label, target

    def __len__(self):
        return self.num_samples
    


def get_dataset(imgList, labelList, resultList):
    # 1. Resize the image to (64, 64)
    # 2. Linearly map [0, 1] to [-1, 1]
    print("get data")
    compose = [
        transforms.ToPILImage(),
        transforms.ColorJitter(brightness=(0.9,0.9000001)),
        transforms.ToTensor(),
    ]
    transform = transforms.Compose(compose)
    dataset = CrypkoDataset(imgList, labelList, resultList, transform)
    return dataset

dataset = get_dataset(imgList, labelList, resultList)

model = FaceMorpher().cuda() #第一次訓練使用這個
model.load_state_dict(torch.load('./checkpoints/face_morpher/face_morpher.pt')) #第N次訓練使用這個，保存提取
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas = (0.9,0.999))
model.train()

# 準備 dataloader, model, loss criterion 和 optimizer
img_dataloader = DataLoader(dataset, batch_size=27, shuffle=True)

n_epoch = 100000
epoch_loss = 0
last_epoch_loss=1

# 主要的訓練過程
print("start training")
while True:
    startTime = time.time()
    for epoch in range(n_epoch):

        epoch_loss = 0
        count = 1
        
        for data, label, target in img_dataloader:  

            img=data
            img = Variable(img).cuda()
            label = Variable(label).cuda()
            target = Variable(target).cuda()
            output1, alpha, color = model(img, label)
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


        if (epoch+1) %1==0:
            localtime1 = time.localtime()
            result_time = time.strftime("%Y%m%d%I%M%p", localtime1)
            os.system('xcopy "D:/talking-head-anime-demo-master/checkpoints/face_morpher" "E:/facemorpher"')
            old_path="E:/facemorpher/face_morpher.pt"
            f_name="E:/facemorpher/face_morpher"+str(result_time)+".pt"
            os.rename(old_path,f_name)
        
    # 訓練完成後儲存 model
    torch.save(model.state_dict(), './checkpoints/face_morpher/face_morpher.pt')
