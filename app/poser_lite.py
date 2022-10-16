import os
import sys

sys.path.append(os.getcwd())

from tkinter import Frame, Label, BOTH, Tk, LEFT, HORIZONTAL, Scale, Button, GROOVE, filedialog, PhotoImage, messagebox

import PIL.Image
import PIL.ImageTk
import numpy
import torch
# from test_app.test_test import test

from util import extract_pytorch_image_from_filelike, rgba_to_numpy_image

import shutil
# from typing_extensions import TypeVarTuple
# import torch
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
from tha.rotator_lite import FaceRotator_lite


class ManualPoserApp:
    def __init__(self,
                 master):
        super().__init__()
        paramName = ["LeftEye", "RightEye", "Mouth", "HeadX", "HeadY", "HeadZ","Brightness*(1+N)"]

        self.master = master

        self.master.title("Manual Poser")

        source_image_frame = Frame(self.master, width=256, height=256)
        source_image_frame.pack_propagate(0)
        source_image_frame.pack(side=LEFT)

        self.source_image_label = Label(source_image_frame, text="Nothing yet!")
        self.source_image_label.pack(fill=BOTH, expand=True)

        control_frame = Frame(self.master, borderwidth=2, relief=GROOVE)
        control_frame.pack(side=LEFT, fill='y')

        self.param_sliders = []
        for param in range(3):
            slider = Scale(control_frame,
                           from_=0,
                           to=1,
                           length=256,
                           resolution=0.001,
                           orient=HORIZONTAL)
            slider.set(0)
            slider.pack(fill='x')
            self.param_sliders.append(slider)

            label = Label(control_frame, text=paramName[param])
            label.pack()
        for param in range(3,6):
            slider = Scale(control_frame,
                           from_=-1,
                           to=1,
                           length=256,
                           resolution=0.001,
                           orient=HORIZONTAL)
            slider.set(0)
            slider.pack(fill='x')
            self.param_sliders.append(slider)
            label = Label(control_frame, text=paramName[param])
            label.pack()
        for param in range(6,7):
            slider = Scale(control_frame,
                        from_=-0.3,
                        to=0.3,
                        length=256,
                        resolution=0.001,
                        orient=HORIZONTAL)
            slider.set(0)
            slider.pack(fill='x')
            self.param_sliders.append(slider)
            label = Label(control_frame, text=paramName[param])
            label.pack()
        # for param in range(6,7):
        #     slider = Scale(control_frame,
        #                     from_=-1,
        #                     to=1,
        #                     length=256,
        #                     resolution=0.01,
        #                     orient=HORIZONTAL)
        #     slider.set(0)
        #     slider.pack(fill='x')
        #     self.param_sliders.append(slider)

        #     label = Label(control_frame, text=paramName[param])
        #     label.pack()

        posed_image_frame = Frame(self.master, width=256, height=256)
        posed_image_frame.pack_propagate(0)
        posed_image_frame.pack(side=LEFT)

        self.posed_image_label = Label(posed_image_frame, text="Nothing yet!")
        self.posed_image_label.pack(fill=BOTH, expand=True)

        self.load_source_image_button = Button(control_frame, text="Load Image ...", relief=GROOVE,
                                               command=self.load_image)
        self.load_source_image_button.pack(fill='x')

        self.pose_size = 6
        self.source_image = None
        self.posed_image = None
        self.current_morph_pose = None
        self.current_rotate_pose = None
        self.last_morph_pose = None
        self.last_rotate_pose = None
        self.needs_update = False

        self.master.after(1000 // 30, self.update_image)

        self.modeln1=FaceMorpher().cuda()
        self.modeln1.load_state_dict(torch.load('checkpoint/face_morpher.pt'))
        self.modeln1.eval()
        self.modeln2=FaceRotator_lite().cuda()
        self.modeln2.load_state_dict(torch.load('checkpoint/rotator_lite.pt'))
        self.modeln2.eval()


    def load_image(self):
        file_name = filedialog.askopenfilename(
            filetypes=[("PNG", '*.png')],
            initialdir="data/illust")
        if len(file_name) > 0:
            image = PhotoImage(file=file_name)
            if image.width() != 256 or image.height() != 256:
                message = "The loaded image has size %dx%d, but we require %dx%d." \
                          % (image.width(), image.height(), 256, 256)
                messagebox.showerror("Wrong image size!", message)
            self.source_image_label.configure(image=image, text="")
            self.source_image_label.image = image
            self.source_image_label.pack()

            self.source_image = file_name
            self.needs_update = True

    def update_pose(self):
        self.current_morph_pose = torch.zeros(3)
        self.current_rotate_pose = torch.zeros(3)
        self.current_brightness = torch.zeros(1)
        for i in range(3):
            self.current_morph_pose[i] = self.param_sliders[i].get()
        for i in range(3, 6):
            self.current_rotate_pose[i-3] = self.param_sliders[i].get()
        self.current_brightness = self.param_sliders[6].get()
        self.current_morph_pose = self.current_morph_pose.unsqueeze(dim=0)
        self.current_rotate_pose = self.current_rotate_pose.unsqueeze(dim=0)

    def update_image(self):
        self.update_pose()
        if (not self.needs_update) and self.last_morph_pose is not None and self.last_rotate_pose is not None and (
                (self.last_morph_pose - self.current_morph_pose).abs().sum().item() < 1e-5)and (
                (self.last_rotate_pose - self.current_rotate_pose).abs().sum().item() < 1e-5)and(
                (abs(self.last_brightness - self.current_brightness)) < 1e-5):
            self.master.after(1000 // 30, self.update_image)
            return
        if self.source_image is None:
            self.master.after(1000 // 30, self.update_image)
            return
        self.last_morph_pose = self.current_morph_pose
        self.last_rotate_pose = self.current_rotate_pose
        self.last_brightness = self.current_brightness

        img=torchvision.io.read_image(self.source_image)
        bn=float(self.param_sliders[6].get())
        print(bn)
        compose = [
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=(bn+1,bn+1.000000001)),
            transforms.ToTensor(),
        ]
        transform=transforms.Compose(compose)
        img=transform(img)
        #morph=torch.tensor([[1,1,0.5]])
        #rotate=torch.tensor([[1,-1,0.5]])
        img=img.unsqueeze(dim=0)

        label=Variable(self.current_morph_pose).cuda()
        label2=Variable(self.current_rotate_pose).cuda()
        img=Variable(img).cuda()

        output1, alpha, color = self.modeln1(img, label)
        resampled, grid_change, grid=self.modeln2(output1,label2)
        unloader = transforms.ToPILImage()

        final_image1 = resampled.cpu().clone()
        final_image1 = unloader(final_image1[0]).save("test/final_image.png") 

        #test(self.source_image, self.current_morph_pose, self.current_rotate_pose)

        posed_image = self.source_image
        image = PhotoImage(file="test/final_image.png")
        #pil_image = PIL.Image.fromarray(numpy.uint8(numpy.rint(numpy_image * 255.0)), mode='RGBA')
        #photo_image = PIL.ImageTk.PhotoImage(image=pil_image)

        self.posed_image_label.configure(image=image, text="")
        self.posed_image_label.image = image
        self.posed_image_label.pack()
        self.needs_update = False

        self.master.after(1000 // 30, self.update_image)
   

if __name__ == "__main__":
    root = Tk()
    app = ManualPoserApp(master=root)
    root.mainloop()
