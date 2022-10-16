from operator import mod
from matplotlib.pyplot import get
from torch.autograd.variable import Variable
from torch.functional import Tensor
from torchvision.models import vgg16,vgg16_bn
from torchvision import models
from torch import nn
import torch
from torch.nn import MSELoss, L1Loss
from torch import Tensor
import numpy as np


class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = models.vgg16(pretrained=True).features
        features[0]=nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # print(features)
        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()
        # self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        # for x in range(16, 23):
        #     self.to_relu_4_3.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        # h = self.to_relu_4_3(h)
        # h_relu_4_3 = h
        out = (h_relu_1_2, h_relu_2_2, h_relu_3_3)
        return out

class perceptual_loss:
    def __init__(self,change,target) :
        self.loss=0
        self.change=change
        self.target=target
    def forword(self):
        model=vgg16(pretrained=True).features
        model[0]=nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        for i in range(16,31):
            del model[16]

        ch=self.change
        tar=self.target
        loss=0
        new_model=get_newm(model,15,15)
        new_model.eval()
        new_model.cuda()
        result=new_model(ch)
        vgg_target=new_model(tar)
        loss=L1Loss()(result,vgg_target)
        torch.cuda.empty_cache()


        new_model=get_newm(model,8,15)
        new_model.eval()
        new_model.cuda()
        result=new_model(ch)
        vgg_target=new_model(tar)
        loss=L1Loss()(result,vgg_target)+loss
        torch.cuda.empty_cache()

        new_model=get_newm(model,3,8)
        new_model.eval()
        new_model.cuda()
        result=new_model(ch)
        vgg_target=new_model(tar)
        loss=L1Loss()(result,vgg_target)+loss
        torch.cuda.empty_cache()

        return loss


def getloss(change,target):
    # model=vgg16(pretrained=True).features
    # model[0]=nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # for i in range(16,31):
    #     del model[16]

    ch=change
    tar=target
    # loss=0
    # new_model=get_newm(model,3,15)
    # new_model.eval()
    # new_model.cuda()
    # result=new_model(ch)
    # vgg_target=new_model(tar)
    # loss=L1Loss()(result,vgg_target)
    # torch.cuda.empty_cache()
    # print(model)
    # print(new_model)


    # new_model=get_newm(model,8,15)
    # new_model.eval()
    # new_model.cuda()
    # result=new_model(ch)
    # vgg_target=new_model(tar)
    # loss=L1Loss()(result,vgg_target)+loss
    # torch.cuda.empty_cache()

    # new_model=get_newm(model,3,8)
    # new_model.eval()
    # new_model.cuda()
    # result=new_model(ch)
    # vgg_target=new_model(tar)
    # loss=L1Loss()(result,vgg_target)+loss
    # torch.cuda.empty_cache()


            
    # return loss*4/3
    nmodel=Vgg16()
    nmodel.eval()
    nmodel.cuda()
    result=nmodel(ch)
    torch.cuda.empty_cache()
    vgg_target=nmodel(tar)
    torch.cuda.empty_cache()
    # print(result)
    # print(vgg_target)
    loss=0
    for i in range(0,3):
        loss=loss+L1Loss()(result[i],vgg_target[i])
    torch.cuda.empty_cache()
    return loss

def get_newm(model,num,last):
    model1=model
    for i in range(num+1,last+1):
        # try:
            del model[num+1]
        # except:
        #     print("out of ramge")
    return model





