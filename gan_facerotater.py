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
# from tensorflow import keras
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

# In[2]:


imgFolder = 'D:/talking-head-anime-demo-master/dataSet1/img1'                     #Img Path
labelFolder = 'D:/talking-head-anime-demo-master/dataSet1/label1'                #Label Path
dataImgTxt = './tryDataImg2.txt'             #Output DataImg.txt
targetImgTxt = './tryTargetImg2.txt'         #Output TargetImg.txt
labelTxt = './tryLabel2.txt'    


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
        return img,target ,label2

    def __len__(self):
        return self.num_samples
    

#此為資料處理函數，經調整亮度後，將圖片轉為Tensor(transform.Compose(compose))
def get_dataset(imgList, labelList, resultList):
    compose = [
        transforms.ToPILImage(),
        #亮度調整
        # transforms.ColorJitter(brightness=(0.92,0.9200001)),
        transforms.ToTensor(),
    ]
    transform = transforms.Compose(compose)
    dataset = CrypkoDataset(imgList, labelList, resultList, transform)
    return dataset

def compose_tensor(input):
    output=torch.tensor([]).to("cuda")
    compose = [
        #亮度調整
        transforms.ToPILImage(),
        # transforms.ColorJitter(brightness=(0.92,0.9200001)),
        
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406,0.406], [0.229, 0.224, 0.225,0.225])
    ]
    transform = transforms.Compose(compose)
    for i in range(int(input.size()[0])):
        # print("output")
        # print(output)
        # print(output.size())
        temp_tensor = transform(input[i]).to("cuda").unsqueeze(0)
        # print(temp_tensor.size())
        # print("temp_tensor")
        # print(temp_tensor)
        # print(temp_tensor.size())
        output=torch.cat((output,temp_tensor),0)
        output=Variable(output).to("cuda")
        

    
    return output


# In[5]:


import torchvision  
print(torchvision.__version__)


# In[6]:

#資料處理，將處理過的東西存入dataset
dataset = get_dataset(imgList, labelList, resultList)



# In[11]:



#讀取和建立TwoAlgoFaceRotator模型
model2= TwoAlgoFaceRotator().cuda()
model2.load_state_dict(torch.load('./two_algo_face_rotator.pt'))
D_model=Discriminator().cuda()
print(D_model)
D_model.load_state_dict(torch.load('./D_model.pt'))
# 


# for name,param in model2.named_parameters():
#     if "zhou_grid_change" in name:
#         print("lock")
#         param.requires_grad = False
#定義loss
criterion = nn.L1Loss()
# criterion = perceptual_loss()
criterion1=nn.BCELoss()
#Adam為一種梯度下降優化演算法
G_optimizer = torch.optim.Adam(model2.parameters(), lr=0.00015, betas = (0.9,0.999))
D_optimizer = torch.optim.Adam(D_model.parameters(), lr=0.00000013, betas = (0.5,0.999))
#0.002 0.000002 is too slow loss 0.5 0.2
#0.002 0.00001 is too high
#0.002 0.000004  is too low loss 0.5 0.25
# 0.0000005


model2.train()
D_model.train()
n_epoch = 5000
start=input("enter start:")
# 準備 dataloader, model, loss criterion 和 optimizer
img_dataloader = DataLoader(dataset, batch_size=20, shuffle=True)

epoch_loss = 0
last_epoch_loss=1
compose2 = [
        # transforms.ToPILImage(),
        #亮度調整
        transforms.ColorJitter(brightness=(0.92,0.9200001)),
        # transforms.ToTensor(),
    ]
transform2 = transforms.Compose(compose2)
# 主要的訓練過程
print("start training")
while True:
    startTime = time.time()
    for epoch in range(int(start),n_epoch):
        epoch_loss = 0
        count = 1
        
        for data, target, label2 in img_dataloader:


            #隨機加入雜訊
            # if random.randint(0,20)<3:
            #     num=random.uniform(0,0.000015)        
            #     img = data + num*torch.randn(256,256)
            # else:
            img=data


            #+是tensor的外包装，也就像錢和錢包那種概念，裡面還有其他東西
            img = Variable(img).cuda()
            label2=Variable(label2).cuda()
            target = Variable(target).cuda()


            #img經過face_morpher處理輸出為output1，其他兩個變數為演算法衍伸物，詳細return內容可見tha/face_morpher.py
            
            #output1經過face_rotator處理輸出為color_changed和resampled，其他兩個變數為演算法衍伸物，詳細return內容和演算法可見tha/two_algo_face_rotator.py
            color_changed, resampled, color_change, alpha_mask, grid_change, grid=model2(img,label2)
            pose = label2.unsqueeze(2).unsqueeze(3)
            pose = pose.expand(pose.size(0), pose.size(1), color_changed.size(2), color_changed.size(3))

            # c1,c2,c3=D_model(cc)
            D_model.eval()
            colorChange=compose_tensor(color_changed)
            c1=D_model(colorChange,pose)
            resampled2=compose_tensor(resampled)
            # r1,r2,r3=D_model(rr)
            r1=D_model(resampled2,pose)
            # print(r1)
            Gan_loss_c=criterion1(c1,Variable(torch.ones(c1.size()).cuda()))
            Gan_loss_r=criterion1(r1,Variable(torch.ones(r1.size()).cuda()))
            Gan_loss=Gan_loss_c+Gan_loss_r
            # Gan_loss=criterion1(c1,torch.ones(c1.size()).cuda())+criterion1(c2,torch.ones(c2.size()).cuda())+criterion1(c3,torch.ones(c3.size()).cuda())+criterion1(r1,torch.ones(r1.size()).cuda())+criterion1(r2,torch.ones(r2.size()).cuda())+criterion1(r3,torch.ones(r3.size()).cuda())
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
            # print(target.size())

            target2=target

            # tt=torch.cat([target,pose],dim=1)
            # t1,t2,t3=D_model(tt)
            if random.randint(0,20)<10:
                num=random.uniform(0,0.000036)        
                target2= target2 + (num*torch.randn(target.size())).cuda()
            target2=compose_tensor(target2)
            t1=D_model(target2,pose)
            # GAN2_loss=criterion1(t1,torch.ones(t1.size()).cuda())+criterion1(t2,torch.ones(t2.size()).cuda())+criterion1(t3,torch.ones(t3.size()).cuda())
            GAN2_loss=criterion1(t1,Variable(torch.ones(t1.size())).cuda())
            a=random.randint(0,20)
            
            # fc=torch.cat([color_changed.detach(),pose],dim=1)
            
            
            # fr=torch.cat([resampled.detach(),pose],dim=1)
            
            # f1,f2,f3=D_model(fc)
            colorChanged = compose_tensor(color_changed.detach())
            f1=D_model(colorChanged,pose)
            # print(f1)
            # print(f2)
            # print(f3)
            # fr1,fr2,fr3=D_model(fr)
            resampled1 = compose_tensor(resampled.detach())
            fr1=D_model(resampled1,pose)
            # GAN3_loss=criterion1(f1,torch.zeros(f1.size()).cuda())+criterion1(f2,torch.zeros(f2.size()).cuda())+criterion1(f3,torch.zeros(f3.size()).cuda())+criterion1(fr1,torch.zeros(fr1.size()).cuda())+criterion1(fr2,torch.zeros(fr2.size()).cuda())+criterion1(fr3,torch.zeros(fr3.size()).cuda())
            GAN3_loss=criterion1(f1,Variable(torch.zeros(f1.size())).cuda())+criterion1(fr1,Variable(torch.zeros(fr1.size())).cuda())
            D_loss=(GAN2_loss+GAN3_loss/2)/2
            D_loss.backward()
            D_optimizer.step()
            torch.cuda.empty_cache()


            #輸出所有演算法衍伸物和結果，以方便觀察(可寫可不寫)
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
            # if count%2000==0:
            #     unloader = transforms.ToPILImage()
            #     original=img.cpu().clone()
            #     original = unloader(original[0]).save("E:/train_file/A_original"+str(epoch)+"_"+str(count)+".png")
            #     target1 = target.cpu().clone()
            #     target1 = unloader(target1[0]).save("E:/train_file/A_target1"+str(epoch)+"_"+str(count)+".png")
            #     color_cd1 = color_changed.cpu().clone()
            #     color_cd1 = unloader(color_cd1[0]).save("E:/train_file/R_color_changed"+str(epoch)+"_"+str(count)+".png")
            #     resampled1 = resampled.cpu().clone()
            #     resampled1 = unloader(resampled1[0]).save("E:/train_file/R_resampled"+str(epoch)+"_"+str(count)+".png")
            #     color_change1 = color_change.cpu().clone()
            #     color_change1 = unloader(color_change1[0]).save("E:/train_file/R_color_change"+str(epoch)+"_"+str(count)+".png")

            #計算epoch的loss之和，並準備下一次訓練
            epoch_loss += pix_loss.item()

            print('epoch:[{}], batch:[{}/{}], G_GANC:[{}], G_GANR:[{}], G_GAN[{}], G_full[{}], G_PIXC[{}], G_PIXR[{}], G_PIXA[{}], D_GAN[{}], D_True[{}], D_Fake[{}], time:[{}:{}]'.format(epoch, count, len(img_dataloader), round(Gan_loss_c.item(),4),round(Gan_loss_r.item(),4),round(Gan_loss.item(),4),round(loss.item(),4),round(pix_c.item(),4),round(pix_r.item(),4), round(pix_loss.item(),4),round(D_loss.item(),4),round(GAN2_loss.item(),4),round((GAN3_loss.item())/2,4),int((time.time()-startTime)/60/60), int((time.time()-startTime)/60%60)))
            count+=1
            if (count+1)%50==0:
                torch.save(model2.state_dict(), './two_algo_face_rotator.pt')
                torch.save(D_model.state_dict(), './D_model.pt')
                print('save successfully')

            #每執行50次進行一次存檔
    #         if count % 50 == 0 :
    #             torch.save(model2.state_dict(), './checkpoints/two_algo_face_rotator/two_algo_face_rotator.pt')
    #             print('save successfully')
            
    #         if count % 6000 ==0:
    #             localtime1 = time.localtime()
    #             result_time = time.strftime("%Y%m%d%I%M%p", localtime1)
    #             os.system('xcopy "D:/talking-head-anime-demo-master/checkpoints/two_algo_face_rotator" "E:/face_rotater"')
    #             old_path="E:/face_rotater/two_algo_face_rotator.pt"
    #             f_name="E:/face_rotater/two_algo_face_rotator_more"+str(result_time)+".pt"
    #             os.rename(old_path,f_name)
    #         if count%500==0:
    #             localtime1 = time.localtime()
    #             result_time = time.strftime("%Y%m%d%I%M%p", localtime1)
    #             inform=str(result_time)+"_"+str(epoch)+"epoch_"
    #             test(str(inform))
    #             switch(result_time)
    #             model2.train()
    #         torch.cuda.empty_cache()
            
    #     #完成一個epoch後進行存檔        
    #     torch.save(model2.state_dict(),"./checkpoints/two_algo_face_rotator/two_algo_face_rotator.pt")
    #     print("save_successfully")

        if (epoch+1)%10==0:
            torch.save(model2.state_dict(), './two_algo_face_rotator.pt')
            torch.save(D_model.state_dict(), './D_model.pt')
            print('save successfully')
    #     if (epoch+1) %60==0:
    #             localtime1 = time.localtime()
    #             result_time = time.strftime("%Y%m%d%I%M%p", localtime1)
    #             os.system('xcopy "D:/talking-head-anime-demo-master/checkpoints/two_algo_face_rotator" "E:/face_rotater"')
    #             old_path="E:/face_rotater/two_algo_face_rotator.pt"
    #             f_name="E:/face_rotater/two_algo_face_rotator_more"+str(result_time)+".pt"
    #             os.rename(old_path,f_name)
    #     #輸出本輪解果
       
    # torch.save(model2.state_dict(),"./checkpoints/two_algo_face_rotator/two_algo_face_rotator.pt")

    # In[ ]:





# %%
