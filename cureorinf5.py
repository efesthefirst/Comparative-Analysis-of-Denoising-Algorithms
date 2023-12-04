# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 03:16:56 2023

@author: Mohammed
"""

import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
from torchvision import transforms
import os
#model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
# or
#%%
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnext101_32x8d', pretrained=True)
#%%
# axes = axes.flatten()
accn=np.zeros((8,750))
accf1=np.zeros((8,750))
accf2=np.zeros((8,750))
vaccn=[]
vaccf1=[]
vaccf2=[]


## noisy files here: 
files=[
       'C:/Users/Mohammed/OneDrive - Georgia Institute of Technology/dip project/results_bm3dd/denoised7.txt',
       'C:/Users/Mohammed/OneDrive - Georgia Institute of Technology/dip project/results_bm3dd/denoised6.txt',
       'C:/Users/Mohammed/OneDrive - Georgia Institute of Technology/dip project/results_bm3dd/denoised5.txt',
       'C:/Users/Mohammed/OneDrive - Georgia Institute of Technology/dip project/results_bm3dd/denoised4.txt',
       'C:/Users/Mohammed/OneDrive - Georgia Institute of Technology/dip project/results_bm3dd/denoised3.txt',
       'C:/Users/Mohammed/OneDrive - Georgia Institute of Technology/dip project/results_bm3dd/denoised2.txt',
       'C:/Users/Mohammed/OneDrive - Georgia Institute of Technology/dip project/results_bm3dd/denoised1.txt',
       'C:/Users/Mohammed/OneDrive - Georgia Institute of Technology/dip project/results_bm3dd/cure_or_orig.txt'
       ]
fdeno1=[
        'C:/Users/Mohammed/OneDrive - Georgia Institute of Technology/denoised_samples/cure_or_orig.txt',
        'C:/Users/Mohammed/OneDrive - Georgia Institute of Technology/denoised_samples/18_grayscale_saltpepper_denoised.txt',
       'C:/Users/Mohammed/OneDrive - Georgia Institute of Technology/denoised_samples/17_grayscale_dirtylens2_denoised.txt',
       'C:/Users/Mohammed/OneDrive - Georgia Institute of Technology/denoised_samples/16_grayscale_dirtylens1_denoised.txt',
       'C:/Users/Mohammed/OneDrive - Georgia Institute of Technology/denoised_samples/15_grayscale_contrast_denoised.txt',
       'C:/Users/Mohammed/OneDrive - Georgia Institute of Technology/denoised_samples/14_grayscale_blur_denoised.txt',
       'C:/Users/Mohammed/OneDrive - Georgia Institute of Technology/denoised_samples/13_grayscale_overexposure_denoised.txt',
       'C:/Users/Mohammed/OneDrive - Georgia Institute of Technology/denoised_samples/12_grayscale_underexposure_denoised.txt',
       
       ]
fdeno2=["E:/denoised7","E:/denoised6","E:/denoised5","E:/denoised4","E:/denoised3","E:/denoised2","E:/denoised1"
       ]
clng=["salt&pepper","dirty lens2","dirty lens1","contrast",
      "blur","overexposure","underexposure","No challenge"]
fdeno1.reverse()

#%%
class dset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data, label,transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = data
        self.transform= transform
        self.label=label
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = Image.open(self.data[idx]).convert('RGB')
        lbl=self.label[idx]
       

        if self.transform:
            image = self.transform(image)

        return image,lbl

preprocess = transforms.Compose([
    
    transforms.Resize(256),
    transforms.CenterCrop(224),
    
    
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

#%%

for p,file in enumerate(files):
    print(p)
    f=open(file,'r')
    ff1=open(fdeno1[p],'r')
    
    data= f.readlines()[1:]
    df1=ff1.readlines()
    
    ## filenames only of noisy images 
    datan=[d.split()[0] for d in data]
    
    labels=[d.split("\\")[-1][6:9] for d in data]
    
    
    labell=[l for l in labels if l=='005' or l== '006' or l=='011' or l=='013' or l=='017' or l== '025' or l=='045']
    datan=[l for l in datan if l.split("\\")[-1][6:9]=='005' or l.split("\\")[-1][6:9]== '006' or l.split("\\")[-1][6:9]=='011' or l.split("\\")[-1][6:9]=='013' or l.split("\\")[-1][6:9]=='017' or l.split("\\")[-1][6:9]== '025' or l.split("\\")[-1][6:9]=='045']
    ## filesnames for denoised images : 
    if p<7:
        dataf1=[d.split()[0] for d in df1]
        dataf1=[l for l in dataf1 if l.split("\\")[-1][6:9]=='005' or l.split("\\")[-1][6:9]== '006' or l.split("\\")[-1][6:9]=='011' or l.split("\\")[-1][6:9]=='013' or l.split("\\")[-1][6:9]=='017' or l.split("\\")[-1][6:9]== '025' or l.split("\\")[-1][6:9]=='045']
        dataf2=[os.path.join(fdeno2[p],*d.split("\\")[2:]) for d in datan]    

    ma=['005', '429','006' ,'811','011', '487', '013', '487','017', '470','025', '837','045', '505']
    
    lab=[ma[ma.index(m)+1] for l in labell for m in ma if m==l]
    
    

    testn=dset(datan,lab,preprocess)
    test_loadern=torch.utils.data.DataLoader(testn,
                                                 batch_size=1, shuffle=False,
                                                 )
    
    testf1=dset(dataf1,lab,preprocess)
    test_loaderf1=torch.utils.data.DataLoader(testf1,
                                                 batch_size=1, shuffle=False,
                                                 )
    
    
    testf2=dset(dataf2,lab,preprocess)
    test_loaderf2=torch.utils.data.DataLoader(testf2,
                                                 batch_size=1, shuffle=False,
                                                 )
    
    model.eval()
    y_predn=np.zeros((len(datan),1000))
    y_predf1=np.zeros((len(dataf1),1000))
    y_predf2=np.zeros((len(dataf2),1000))
    with torch.no_grad() : 
        for i, (section, label) in enumerate(test_loadern):
            y_predn[i][:]=model(section)
        if p<7: 
            for i, (section, label) in enumerate(test_loaderf1):
                y_predf1[i][:]=model(section)
            for i, (section, label) in enumerate(test_loaderf2):
                y_predf2[i][:]=model(section)
    
    y_predn=torch.nn.functional.softmax(torch.tensor(y_predn), dim=0)
    tobn, tidn= torch.topk(y_predn, 10)
    
    y_predf1=torch.nn.functional.softmax(torch.tensor(y_predf1), dim=0)
    tobf1, tidf1= torch.topk(y_predf1, 10)
    
    y_predf2=torch.nn.functional.softmax(torch.tensor(y_predf2), dim=0)
    tobf2, tidf2= torch.topk(y_predf2, 10)
    
    tidn=tidn.numpy()
    tidf1=tidf1.numpy()
    
    
    tidf2=tidf2.numpy()
    for i,l in enumerate(lab): 
        if int(l) in tidn[i]:
            accn[p,i]=1
        else:
            accn[p,i]=0
    for i,l in enumerate(lab): 
        if int(l) in tidf1[i]:
            accf1[p,i]=1
        else:
            accf1[p,i]=0
            
            
    for i,l in enumerate(lab): 
        if int(l) in tidf2[i]:
            accf2[p,i]=1
        else:
            accf2[p,i]=0
    if p!=7:
        vaccn.append(sum(accn[p])/750*100)
        vaccf1.append(sum(accf1[p])/750*100)
        vaccf2.append(sum(accf2[p])/750*100)
    else:
       vaccn.append(sum(accn[p])/149*100)
       vaccf1.append(sum(accf1[p])/750*100)
    # ax=axes[p]
    # ax.plot(clng[p],acc[p], marker='o', label="BM3D-Net Denoising")
    # ax.set_title(f'Top 10 Accuracy')
    # ax.set_xlabel('Background')
#%%
plt.plot(clng[-1],vaccn[-1],marker='o',label="Top 10 Noise-free images accuracy")
plt.plot(clng[:-1],vaccn[:-1],marker='o',label="Top 10 Noisy images accuracy")
plt.plot(clng[:-1],vaccf2[:7],marker='o',label="Top 10 BM3D Denoised images accuracy")
plt.plot(clng[:-1],vaccf1[:7],marker='o',label="Top 10 FFDNet Denoised images accuracy")
plt.title("Top 10 Accuracy on Cure-OR dataset")
plt.xlabel("Challenge")
plt.ylabel("Accuracy")
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.savefig('C:/Users/Mohammed/OneDrive - Georgia Institute of Technology/dip project/cureor_class1.png',dpi=300)