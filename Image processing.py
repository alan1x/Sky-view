# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 12:29:38 2023

@author: rcorr
"""

import matplotlib.pyplot as plt
import matplotlib.image as mli
import numpy as np
import numpy.ma as ma
from scipy.ndimage import gaussian_filter
plt.style.use('default')
imname='Team2Loc2.png'
myimage=mli.imread(imname)
plt.figure()
plt.subplot(3,2,1)
plt.imshow(myimage[:,:,1],cmap='copper')
plt.title('Hopper 1940')
mp=['Reds','Greens','Blues','Greys']
t=['Red component','Green component','Blue component']
for i in range(3):
    plt.subplot(3,2,i+2)
    plt.imshow(myimage[:,:,i],cmap=mp[i])
    plt.title(t[i])

grey=.2126*myimage[:,:,0]+.7152*myimage[:,:,1]+.0722*myimage[:,:,2]
greyf=np.ndarray.flatten(grey)
greyf_m=ma.masked_less(grey, 0.3)
greyf_mf=np.ndarray.flatten(greyf_m)
plt.subplot(3,2,5)
plt.imshow(grey,cmap='Greys_r')
plt.title('Greys component')    
plt.subplot(3,2,6)
plt.hist(greyf,bins=100)
plt.title('Greys hist')      
plt.tight_layout()

### Greys
plt.figure()

plt.subplot(2,2,1)
plt.imshow(grey,cmap='Greys_r')
plt.title('Greys component')   
plt.tight_layout() 
plt.subplot(2,2,2)
plt.hist(greyf,bins=100)
plt.title('Greys hist')  
plt.tight_layout()   
plt.subplot(2,2,3)
plt.imshow(greyf_m,cmap='Greys_r')
plt.title('Greys mask')
plt.tight_layout()
plt.subplot(2,2,4)
plt.hist(greyf_mf,bins=100)
plt.title('Greys mask hist') 
plt.tight_layout()





for i in range(7):
    imname=f'Team2Loc{i+1}.png'

    Loc_ori=mli.imread(imname) 
    Npixels=np.size(Loc_ori[:,:,0])

    Ncolors = 3 
    gaufil = 1

    if gaufil==0: 
        Loc = Loc_ori

    elif gaufil==1: 
        sigma=20
        Loc= np.empty_like(Loc_ori)
        for k in range(Ncolors):
             Loc[:,:,k]=gaussian_filter(Loc_ori[:,:,k], sigma)
    th = 0.2         
    plt.figure()
    for i in range(3):
        plt.subplot(3, 3, i*3 + 1)
        plt.imshow(Loc[:, :, i], cmap='copper')
        plt.title(t[i])

        plt.subplot(3, 3, i*3 + 2)
        greyf_mf = np.ndarray.flatten(Loc[:, :, i])
        plt.hist(greyf_mf, bins=100)
        plt.axvline(th,0,color='black')
        plt.title(f'{t[i]} Histogram')
        
        plt.subplot(3, 3, i*3 + 3)
        sfp = ma.masked_less(Loc[:, :, i],th)
        sf = np.sum(Loc[:,:,i]>th)
        bs = np.size(Loc[:, :, i])
        skv = round(100*(sf/bs),4)
        plt.imshow(sfp,cmap='copper')
        plt.title(f' Sky view Factor {skv}')
    plt.tight_layout()
    plt.show()










