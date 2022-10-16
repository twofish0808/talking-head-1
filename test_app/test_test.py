import os
import sys
sys.path.append(os.getcwd())
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
import pandas as pd


def test(img, morph, rotate):
    model = FaceMorpher().cuda() 
    model.load_state_dict(torch.load('checkpoint/face_morpher.pt')) 

    #讀取和建立TwoAlgoFaceRotator模型
    modeln2= TwoAlgoFaceRotator().cuda()
    modeln2.load_state_dict(torch.load('checkpoint/two_algo_face_rotator.pt'))


    modeln3=Combiner().cuda()
    modeln3.load_state_dict(torch.load('checkpoint/combiner.pt'))

    model.eval()
    modeln2.eval()
    modeln3.eval()

    #img="./ki.png"
    img=torchvision.io.read_image(img)


    compose = [
        transforms.ToPILImage(),
        transforms.ToTensor(),
            ]
    transform=transforms.Compose(compose)
    img=transform(img)

    #morph=torch.tensor([[1,1,0.5]])
    #rotate=torch.tensor([[1,-1,0.5]])
    img=img.unsqueeze(dim=0)

    label=Variable(morph).cuda()
    label2=Variable(rotate).cuda()
    img=Variable(img).cuda()

    output1, alpha, color = model(img, label)
    color_changed, resampled, color_change, alpha_mask, grid_change, grid=modeln2(output1,label2)
    final_image, combined_image, combine_alpha_mask, retouch_alpha_mask, retouch_color_change=modeln3(color_changed,resampled,label2)
    unloader = transforms.ToPILImage()

    final_image1 = final_image.cpu().clone()
    final_image1 = unloader(final_image1[0]).save("test/final_image.png") 


    
    print("test.py")
    torch.cuda.empty_cache()


if __name__ == '__main__':
    img="ki.png"
    morph=torch.tensor([[1,1,0.5]])
    rotate=torch.tensor([[1,-1,0.5]])
    test(img, morph, rotate)