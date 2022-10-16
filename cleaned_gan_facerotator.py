import os
import shutil
from unicodedata import name
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
from tha.perceptual import perceptual_loss,getloss, vgg_discriminator,Discriminator,Vgg_discriminator
from tha.face_morpher import FaceMorpher
from tha.two_algo_face_rotator import TwoAlgoFaceRotator
from tha.dis import NetD


from time_controler.switch import switch
from test import test
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(torch.cuda.current_device()))



imgFolder = 'D:/talking-head-anime-demo-master/dataSet1/img1'                     #Img Path
labelFolder = 'D:/talking-head-anime-demo-master/dataSet1/label1'                #Label Path
dataImgTxt = './tryDataImg2.txt'             #Output DataImg.txt
targetImgTxt = './tryTargetImg2.txt'         #Output TargetImg.txt
labelTxt = './tryLabel2.txt'    



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
        img = torchvision.io.read_image(imgList)
        img = self.transform(img)
        resultList = self.resultList[idx]
        target = torchvision.io.read_image(resultList)
        target = self.transform(target)        
        label2=self.labelList[idx].split(',')[3:]
        label2 = [float(i) for i in label2]
        label2= torch.FloatTensor(label2)
        label2=torch.div(label2,15)
        return img,target ,label2

    def __len__(self):
        return self.num_samples
    
def get_dataset(imgList, labelList, resultList):
    compose = [
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ]
    transform = transforms.Compose(compose)
    dataset = CrypkoDataset(imgList, labelList, resultList, transform)
    return dataset

def compose_tensor(input):
    output=torch.tensor([]).to("cuda")
    compose = [
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406,0.406], [0.229, 0.224, 0.225,0.225])
    ]
    transform = transforms.Compose(compose)
    for i in range(int(input.size()[0])):
        temp_tensor = transform(input[i]).to("cuda").unsqueeze(0)
        output=torch.cat((output,temp_tensor),0)
        output=Variable(output).to("cuda")
        
    return output

print(torchvision.__version__)



dataset = get_dataset(imgList, labelList, resultList)
model2= TwoAlgoFaceRotator().cuda()
model2.load_state_dict(torch.load('./two_algo_face_rotator.pt'))
D_model=Discriminator().cuda()
print(D_model)
D_model.load_state_dict(torch.load('./D_model.pt'))
criterion = nn.L1Loss()
criterion1=nn.BCELoss()
G_optimizer = torch.optim.Adam(model2.parameters(), lr=0.00015, betas = (0.9,0.999))
D_optimizer = torch.optim.Adam(D_model.parameters(), lr=0.00000013, betas = (0.5,0.999))


model2.train()
D_model.train()
n_epoch = 5000
start=input("enter start:")
img_dataloader = DataLoader(dataset, batch_size=20, shuffle=True)

epoch_loss = 0
last_epoch_loss=1
compose2 = [
        transforms.ColorJitter(brightness=(0.92,0.9200001)),
    ]
transform2 = transforms.Compose(compose2)
print("start training")
while True:
    startTime = time.time()
    for epoch in range(int(start),n_epoch):
        epoch_loss = 0
        count = 1
        
        for data, target, label2 in img_dataloader:

            img=data
            img = Variable(img).cuda()
            label2=Variable(label2).cuda()
            target = Variable(target).cuda()
            color_changed, resampled, color_change, alpha_mask, grid_change, grid=model2(img,label2)
            pose = label2.unsqueeze(2).unsqueeze(3)
            pose = pose.expand(pose.size(0), pose.size(1), color_changed.size(2), color_changed.size(3))
            D_model.eval()
            colorChange=compose_tensor(color_changed)
            c1=D_model(colorChange,pose)
            resampled2=compose_tensor(resampled)
            r1=D_model(resampled2,pose)
            Gan_loss_c=criterion1(c1,Variable(torch.ones(c1.size()).cuda()))
            Gan_loss_r=criterion1(r1,Variable(torch.ones(r1.size()).cuda()))
            Gan_loss=Gan_loss_c+Gan_loss_r
            pix_c=criterion(color_changed,target)
            pix_r=criterion(resampled,target)
            pix_loss=pix_c+pix_r
            loss=(pix_loss*90+Gan_loss)/60
            G_optimizer.zero_grad()
            loss.backward()
            G_optimizer.step()
            torch.cuda.empty_cache()

            D_model.train()
            D_optimizer.zero_grad()
            target2=target
            if random.randint(0,20)<10:
                num=random.uniform(0,0.000036)        
                target2= target2 + (num*torch.randn(target.size())).cuda()
            target2=compose_tensor(target2)
            t1=D_model(target2,pose)
            GAN2_loss=criterion1(t1,Variable(torch.ones(t1.size())).cuda())
            a=random.randint(0,20)
            colorChanged = compose_tensor(color_changed.detach())
            f1=D_model(colorChanged,pose)
            resampled1 = compose_tensor(resampled.detach())
            fr1=D_model(resampled1,pose)
            GAN3_loss=criterion1(f1,Variable(torch.zeros(f1.size())).cuda())+criterion1(fr1,Variable(torch.zeros(fr1.size())).cuda())
            D_loss=(GAN2_loss+GAN3_loss/2)/2
            D_loss.backward()
            D_optimizer.step()
            torch.cuda.empty_cache()


            try:
            
                unloader = transforms.ToPILImage()
                original=img.cpu().clone()
                original = unloader(original[0]).save("D:/train_file/A_original.png")
                target1 = target.cpu().clone()
                target1 = unloader(target1[0]).save("D:/train_file/A_target1.png")
                color_cd1 = color_changed.cpu().clone()
                color_cd1 = unloader(color_cd1[0]).save("D:/train_file/R_color_changed.png")
                resampled12 = resampled.cpu().clone()
                resampled12 = unloader(resampled12[0]).save("D:/train_file/R_resampled.png")
                color_change12 = color_change.cpu().clone()
                color_change12 = unloader(color_change12[0]).save("D:/train_file/R_color_change.png")
                aa=resampled1.cpu().clone()
                aa=unloader(aa[0]).save("D:/train_file/R_compose_resampled.png")
                bb=colorChanged.cpu().clone()
                bb=unloader(bb[0]).save("D:/train_file/R_compose_color_change.png")
                cc=target2.cpu().clone()
                cc=unloader(cc[0]).save("D:/train_file/R_compose_target.png")
                
            except:
                pass

            epoch_loss += pix_loss.item()

            print('epoch:[{}], batch:[{}/{}], G_GANC:[{}], G_GANR:[{}], G_GAN[{}], G_full[{}], G_PIXC[{}], G_PIXR[{}], G_PIXA[{}], D_GAN[{}], D_True[{}], D_Fake[{}], time:[{}:{}]'.format(epoch, count, len(img_dataloader), round(Gan_loss_c.item(),4),round(Gan_loss_r.item(),4),round(Gan_loss.item(),4),round(loss.item(),4),round(pix_c.item(),4),round(pix_r.item(),4), round(pix_loss.item(),4),round(D_loss.item(),4),round(GAN2_loss.item(),4),round((GAN3_loss.item())/2,4),int((time.time()-startTime)/60/60), int((time.time()-startTime)/60%60)))
            count+=1
            if (count+1)%50==0:
                torch.save(model2.state_dict(), './two_algo_face_rotator.pt')
                torch.save(D_model.state_dict(), './D_model.pt')
                print('save successfully')

        if (epoch+1)%50==0:
            torch.save(model2.state_dict(), './two_algo_face_rotator.pt')
            torch.save(D_model.state_dict(), './D_model.pt')
            print('save successfully')

