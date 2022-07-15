# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 09:58:14 2022

@author: pheu
"""

import os,  h5py
import numpy as np

import time

import matplotlib.pyplot as plt

from invertdeflectpy import root_dir
from invertdeflectpy.pic import InvertPIC

# TODO: Add a test with a non-uniform I0


def test_1d():
    
    file = os.path.join(root_dir, 'examples',
                        'gaussian_charge_cylinder_mu_0p5_highres.h5')
    
    with h5py.File(file, 'r') as f:
        img = f['image'][:]
        
    # Create a 1D lineout by averaging over a slice of this cylindrical
    # 2D radiograph
    x = np.linspace(-1,1, img.shape[0])
    I = np.mean(img[:, 30:120], axis=-1)
    
    obj = InvertPIC(I, N_cell_min=50, tolerance=1e-4)
    obj.run()
    
    I_rec, F = obj.results
    
    fig, ax = plt.subplots()
    ax.plot(x, I, label='Radiograph')
    ax.plot(x, I_rec, label='Reconstruction')
    ax.plot(x, F, label='Fdl')
    ax.legend()
    
def test_2d():

    plots=True
    
    file = os.path.join(root_dir, 'examples',
                        'gaussian_charge_blob_mu_0p5_veryhighres_lowres.h5')
    
    with h5py.File(file, 'r') as f:
        img = f['image'][:]
        
        
    x = np.linspace(-1,1, img.shape[0])
    y = np.linspace(-1,1, img.shape[1])
    I = img

    if plots:
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.pcolormesh(x, y,  I.T)
        ax.set_title("Radiograph")
        plt.show()
    
    
    t0 = time.time()
    
    obj = InvertPIC(I, N_cell_min=4, tolerance=1e-2, max_iter=60,
                        plots=plots, verbose=True)
    obj.run()
    
    
    t1 = time.time()
    print(f"Execution time: {t1-t0:.1f} s")
    
    
    if plots:
        I_rec, Fx, Fy = obj.results
        
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.pcolormesh(x, y,  I_rec.T)
        ax.set_title("Reconstruction")
        
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.pcolormesh(x, y,  Fx.T)
        ax.set_title("Fxdl")
        
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.pcolormesh(x, y,  Fy.T)
        ax.set_title("Fydl")
        
        fig, ax = plt.subplots()
        ax.set_xlabel("Iteration")
        iterations = np.arange(len(obj.energies))
        ax.plot(iterations, obj.energies, label='Energy')
        ax.axhline(obj.energy_stop, label='Energy stop', color='black')
        ax.set_title("Energy")
        ax.legend(loc='upper right')
        
        plt.show()
    

    
    

    
        
if __name__ == "__main__":
    test_1d()
    test_2d()
    

