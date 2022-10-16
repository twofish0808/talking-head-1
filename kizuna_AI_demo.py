import os
import shutil
# from typing_extensions import TypeVarTuple
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
from tha.face_morpher import FaceMorpher
from tha.two_algo_face_rotator import TwoAlgoFaceRotator
from tha.combiner import Combiner


model = FaceMorpher().cuda() 
model.load_state_dict(torch.load('./checkpoints/face_morpher/face_morpher.pt')) 

#讀取和建立TwoAlgoFaceRotator模型
model2= TwoAlgoFaceRotator().cuda()
model2.load_state_dict(torch.load('./checkpoints/two_algo_face_rotator/two_algo_face_rotator.pt'))


model3=Combiner().cuda()
model3.load_state_dict(torch.load('./checkpoints/combiner/combiner.pt'))

model.eval()
model2.eval()
model3.eval()

# print("please enter model number")
# model_num=str(input())
# img="D:/talking-head-anime-demo-master/dataSet1/img/"+model_num+"/MMD"+model_num+"_0.png"

# img="C:/Users/twofi/Desktop/hotaru.png"
img="C:/Users/twofi/Desktop/ki.png"
# img="D:/talking-head-anime-demo-master/data/illust/waifu_00_256.png"
img=torchvision.io.read_image(img)

# label=str(input("morph label Ex:[ 0, 0, 0]"))
# label2=str(input("rotate label Ex:[ 0, 0, 0]"))
# print(label)
# print(label2)

compose = [
    transforms.ToPILImage(),
    #transforms.Resize((64, 64)),
    # transforms.ColorJitter(brightness=(0.9,0.9000001)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
transform=transforms.Compose(compose)
img=transform(img)
# print(type(label))
morph=torch.tensor([[0,1,0.5]])
rotate=torch.tensor([[0,1,0.5]])
img=img.unsqueeze(dim=0)
label=Variable(morph).cuda()
label2=Variable(rotate).cuda()
img=Variable(img).cuda()
output1, alpha, color = model(img, label)
color_changed, resampled, color_change, alpha_mask, grid_change, grid=model2(output1,label2)
final_image, combined_image, combine_alpha_mask, retouch_alpha_mask, retouch_color_change=model3(color_changed,resampled,label2)
unloader = transforms.ToPILImage()
original=img.cuda().clone()
original = unloader(original[0]).save("./test/original.png")
color1 = color.cuda().clone()
color1 = unloader(color1[0]).save("./test/color.png")
alpha1 = alpha.cuda().clone()
alpha1 = unloader(alpha1[0]).save("./test/alpha.png")
output_image = output1.cuda().clone()
output_image = unloader(output_image[0]).save("./test/output_image.png")
color_cd1 = color_changed.cuda().clone()
color_cd1 = unloader(color_cd1[0]).save("./test/color_changed.png")
resampled1 = resampled.cuda().clone()
resampled1 = unloader(resampled1[0]).save("./test/resampled.png")
color_change1 = color_change.cuda().clone()
color_change1 = unloader(color_change1[0]).save("./test/color_change.png")
alpha_mask1 = alpha_mask.cuda().clone()
alpha_mask1 = unloader(alpha_mask1[0]).save("./test/alpha_mask.png")
final_image1 = final_image.cuda().clone()
final_image1 = unloader(final_image1[0]).save("./test/final_image.png") 
combined_image1 = combined_image.cuda().clone()
combined_image1 = unloader(combined_image1[0]).save("./test/combined_image.png")  
combine_alpha_mask1 = combine_alpha_mask.cuda().clone()
combine_alpha_mask1 = unloader(combine_alpha_mask1[0]).save("./test/combine_alpha_mask.png")     
retouch_alpha_mask1 = retouch_alpha_mask.cuda().clone()
retouch_alpha_mask1 = unloader(retouch_alpha_mask1[0]).save("./test/alpha_mask.png")     
retouch_color_change1 = retouch_color_change.cuda().clone()
retouch_color_change1 = unloader(retouch_color_change1[0]).save("./test/alpha_mask.png")