# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 15:01:47 2022

@author: pheu
"""

import os, h5py
from invertdeflectpy import root_dir
import time

import numpy as np
import matplotlib.pyplot as plt

from invertdeflectpy.mongeampere import InvertMA

        
def test_1d():
    file = os.path.join(root_dir, 'examples',
                        'gaussian_charge_cylinder_mu_0p5_highres.h5')
    
    with h5py.File(file, 'r') as f:
        img = f['image'][:]
        
    # Create a 1D lineout by averaging over a slice of this cylindrical
    # 2D radiograph
    x = np.linspace(-1,1, img.shape[0])
    I = np.mean(img[:, 30:120], axis=-1)
    
    obj = InvertMA(I, max_iter=5000, fdt=0.1)
    obj.run()
    
    I_rec, Fx, Fy, dtMS = obj.results
    
    fig, ax = plt.subplots()
    ax.set_title('1D test')
    ax.plot(x, I, label='Radiograph')
    ax.plot(x, I_rec, label='Reconstruction')
    ax.plot([], color='red', label='Fdl')
    ax2 = ax.twinx()
    ax2.plot(x, Fx, label='Fdl', color='red')
    ax2.set_ylabel("Fdl")
    ax.legend()
    
    
def test_2d():
    file = os.path.join(root_dir, 'examples',
                        'gaussian_charge_blob_mu_0p5_veryhighres_lowres.h5')
    
    with h5py.File(file, 'r') as f:
        img = f['image'][:]
        
        
    x = np.linspace(-1,1, img.shape[0])
    y = np.linspace(-1,1, img.shape[1])
    I = img

    fig, axarr = plt.subplots(nrows=2, ncols=2, figsize=(9,9))
    ax = axarr[0][0]
    ax.set_aspect('equal')
    ax.pcolormesh(x, y,  I.T)
    ax.set_title("Input")
    
    
    t0 = time.time()
    
    obj = InvertMA(I, max_iter=5e3, verbose=True, tolerance=1e-3)
    obj.run()
    
    
    t1 = time.time()
    print(f"Execution time: {t1-t0:.1f} s")
    
    
    I_rec, Fx, Fy, dphi_dtMS, dt = obj.results
    
    ax = axarr[0][1]
    ax.set_aspect('equal')
    ax.pcolormesh(x, y,  I_rec.T)
    ax.set_title("Reconstruction")
    
    ax = axarr[1][0]
    ax.set_aspect('equal')
    ax.pcolormesh(x, y,  Fx.T)
    ax.set_title("Fxdl")
    
    ax = axarr[1][1]
    ax.set_aspect('equal')
    ax.pcolormesh(x, y,  Fy.T)
    ax.set_title("Fydl")
    
    fig, ax = plt.subplots()
    ax.set_xlabel("Iteration")
    iterations = np.arange(len(dphi_dtMS))
    ax.plot(iterations, dphi_dtMS, label='dphi_dtMS')
    ax.axhline(obj.tolerance, label='Tolerance', color='black', linestyle='dashed')
    ax.plot([], color='blue', label ='dt')
    ax.set_ylabel("dphi_dtMS")
    ax.legend(loc='upper right')
    
    ax2 = ax.twinx()
    ax2.plot(iterations, dt, label='dt', color='blue')
    ax2.set_ylabel("dt")
    
    
    
if __name__ == '__main__':
    test_1d()
    test_2d()
    