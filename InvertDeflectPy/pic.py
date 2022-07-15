# -*- coding: utf-8 -*-

import os, h5py
import numpy as np
from scipy.interpolate import interpn
from scipy.signal import convolve2d

import time
import matplotlib.pyplot as plt

from numba import njit


@njit
def _weight_1d(x, wf, Nx):
    """
    Linear weighting particles to the nearest grid points
    
    Each particle contributes to the two nearest grid points, proportionally
    to its distance from each grid point.
    
    This is sort of like the opposite of an interpolator, which determines
    the value of grid quantities on each particle.
    
    Variables
    ---------
    
    x: np.ndarray, [N_tot]
        Array of particle positions
        
    wf: np.ndarray, [N_tot]
        Array of weights for each particle
        
    Nx : int
        The number of x points on the grid
        
    """
    # TODO: Validate that this is volume averaging correctly - maybe write a test
    
    f=np.zeros(Nx)
    for i in range(Nx):
        # Linear weight is 
        # weight function * distance of particle from grid point
        x_dist = np.abs(x - (i+0.5)) # i+0.5 is the grid point location
        close = x_dist <= 1
        f[i] = np.sum(wf[close]*(1-x_dist[close]))
 
    return f


@njit(parallel=True)
def _weight_2d(x, y, wf, Nx, Ny):
    
    """
    Linear weighting particles to the nearest grid points
    
    Each particle contributes to the two nearest grid points, proportionally
    to its distance from each grid point.
    
    This is sort of like the opposite of an interpolator, which determines
    the value of grid quantities on each particle.
    
    Variables
    ---------
    
    x: np.ndarray [N_tot]
        Array of particle positions
        
    y: np.ndarray, [N_tot]
        Array of particle positions
        
    wf: np.ndarray, [N_tot]
        Array of weights for each particle
        
    Nx : int
        The number of x points on the grid
        
    Ny : int
        The number of y points on the grid
        
    """
    f=np.zeros((Nx,Ny))
    
    # Nearest integer gives grid point to particle's lower left?? corner 
    # since dx,dy=1
    i,j=np.rint(y), np.rint(x)
    #Loop over the grid and sum up contributions from nearby particles
    
    # range(1, Nx+1) = 1:Nx in matlab
    for ig in range(1, Ny+1):
        for jg in range(1, Nx+1):
            # Lower left corner??
            ip=(i==ig)&(j==jg)
            f[jg-1, ig-1] += np.sum(wf[ip]*(0.5 -x[ip] + jg)*(0.5 -y[ip] + ig))
            # Upper left??
            ip=(i==ig-1)&(j==jg)
            f[jg-1, ig-1] += np.sum(wf[ip]*(0.5 -x[ip] + jg)*(1.5 +y[ip] - ig))
            # Lower right??
            ip=(i==ig)&(j==jg-1)
            f[jg-1, ig-1] += np.sum(wf[ip]*(1.5 +x[ip] - jg)*(0.5 -y[ip] + ig))
            # Upper right??
            ip=(i==ig-1)&(j==jg-1)
            f[jg-1, ig-1] += np.sum(wf[ip]*(1.5 +x[ip] - jg)*(1.5 +y[ip] - ig))
       
    return f


class InvertPIC:
    
    def __init__(self, I : np.ndarray, 
                 I0: np.ndarray =None, 
                 tolerance: float =None, 
                 N_cell_min: int =None,
                 N_cell_max: int =None,
                 max_iter: int =None,
                 plots=False,
                 verbose=True):
        
        self.I = I
        self.plots = plots
        self.verbose = verbose
        
        if self.I.ndim == 1:
            self.twoD = False
        elif self.I.ndim == 2:
            self.twoD = True
        else:
            raise ValueError("I must be either 1-D or 2-D, but provided aray "
                             f"has shape {I.shape}")
        
        if I0 is None:
            print("I0 not specified: assuming a uniform I0.")
            self.I0 = np.ones(I.shape)
        else:
            self.I0 = np.array(I0)
        
        if self.I0.shape != self.I.shape:
            raise ValueError(f"The shape of I0 ({self.I0.shape}) does not match "
                             f" the shape of I ({self.I.shape}).")
            
        if tolerance is None:
            self.tolerance=1e-3
        else:
            self.tolerance = float(tolerance)
            
        if self.tolerance >=1 or self.tolerance <= 0:
            raise ValueError(f"Tolerance ({self.tolerance}) is out of the " 
                             "allowed range (0,1)")
            
        if N_cell_min is None:
            self.N_cell_min = 8
            print("N_cell_min not specified: using "
                  f"the default value ({self.N_cell_min})")
        else:
            self.N_cell_min = int(N_cell_min)
            
        if self.N_cell_min < 1:
            raise ValueError(f"N_cell_min ({self.N_cell_min}) must be > 1.")
            
        
        if N_cell_max is None:
            self.N_cell_max = 120
            print("N_cell_max not specified: using "
                  f"the default value ({self.N_cell_max})")
        else:
            self.N_cell_max = int(N_cell_max)
            
        if self.N_cell_max < 1:
            raise ValueError(f"N_cell_max ({self.N_cell_max}) must be > 1.")
            
        if max_iter is None:
            self.max_iter = 1000
            print("max_iter not specified: using "
                  f"the default value ({self.max_iter})")
        else:
            self.max_iter = int(max_iter)
            
        
        # Setup the simulation  variables
        if self.twoD:
            self._setup_2d()
        else:
            self._setup_1d()
            
            
    def _setup_1d(self):
        #Determine fixed ion density
        self.n_i=self.I/np.mean(self.I)
        
        # Determine number of particles per cell
        # Ensure ability to reproduce lowest n_i with one particle in a cell or by
        # the movement of 1 particle from the cell
        nmin=np.min(self.n_i[self.n_i>0])
        self.N_cell=int(np.round(1/min([nmin,1-nmin]))) # Nearest integer
        self.N_cell=max([self.N_cell_min,self.N_cell]) # Enforce N_cell_min condition
        self.N_cell=min([self.N_cell_max,self.N_cell]) # Enforce N_cell_max condition
        print(f"Number of particles per cell: {self.N_cell:.1e}")
        
        #Number of grid points and total number of particles
        self.Nx = self.n_i.size + 2 # Include two ghost cells 
        self.N_tot = self.Nx * self.N_cell
        
        # Positions of the edge of the bins, grid spacing dz = 1 by definition
        self.x_grid = np.arange(0.5, self.Nx + 0.5)
        
        # Particle positions, velocities and weight functions for electrons
        # Uniformly spaced particles in (0 N_grid) 
        self.x=np.arange(self.N_tot)*self.Nx/(self.N_tot-1)
        # electrons initially at rest
        self.vx=np.zeros(self.N_tot)
        
        # weight particles to produce the required n_e using linear interpolation
        # Pad I0 with 2 constant values on either side
        I0_pad = np.pad(self.I0, (2,2), mode='edge')
        # Create an array of grid left edges, extended to match
        z_grid_left = np.arange(-1, self.Nx+1) # Left edges of the grid cells
        self.wf =interpn((z_grid_left,), I0_pad, (self.x,),
                        method='linear')
        
        # Remove particles with wf=0 (in case I0 has holes in it, eg a grid)
        self.x=self.x[self.wf>0] 
        self.vx=self.vx[self.wf>0] 
        self.wf=self.wf[self.wf>0]
        self.N_tot=self.x.size
        print(f"Total number of particles {self.N_tot:.1e}")
        
        # Store initial position to calculate displacement
        self.x0= np.copy(self.x)
        
     
        self._weight_n_e()
        
        # Ensure the mean of the electron density excluding the ghost cells is 1
        norm=np.mean(self.n_e[1:-1])
        self.n_e *= 1/norm
        self.wf *= 1/norm
        
        # Reconstructed I0 (remove ghost cells and undo normalization of initial n_e)
        self.I0_rec=self.n_e[1:-1]*np.mean(self.I0)
        # Save a copy of n_e to normalize the resulting force with
        self.n0 = np.copy(self.n_e)
        
        #Add ghost cells to n_i that maintain charge neutrality n_i=n_e
        self.n_i= np.pad(self.n_i, (1,1), mode='constant',
                         constant_values=((self.n_e[0], self.n_e[-1])))
        
        # Drag coefficient on the grid set to twice the plasma frequency when
        # n_e=n_i to give critical damping at desired final state
        self.drag = 2*np.sqrt(self.n_i)
        
        self._calculate_E()
           
    def _setup_2d(self):
        self.n_i=self.I/np.mean(self.I)
        self.n_e = self.I0/np.mean(self.I0)
        
        # Add ghost cells to n_e (continuing constant edge values)
        self.n_e = np.pad(self.n_e, (1,1), mode='edge')
        
        # Add ghost cells to n_i that equal the ghost cells in n_e
        self.n_i = np.pad(self.n_i, (1,1), mode='empty')
        self.n_i[0,:] = self.n_e[0,:]
        self.n_i[-1,:] = self.n_e[-1,:]
        self.n_i[:,0] = self.n_e[:,0]
        self.n_i[:,-1] = self.n_e[:,-1]
        
        # Number of grid points
        self.Nx, self.Ny = np.array(self.n_i.shape)
        
        # Determine number of particles per cell
        # Ensure ability to reproduce lowest n_i with one particle in a cell or by
        # the movement of 1 particle from the cell
        nmin=np.min(self.n_i[self.n_i>0])
        self.N_cell=int(np.round(1/np.sqrt(min([nmin,1-nmin])))) # Nearest integer
        self.N_cell=max([self.N_cell_min,self.N_cell]) # Enforce N_cell_min condition
        self.N_cell=min([self.N_cell_max,self.N_cell]) # Enforce N_cell_max condition
        print(f"Number of particles per cell: {self.N_cell**2:.1e}")
        
        # Initial particle positions and velocities
        # Ranging from just inside zero to just inside Nx or Ny
        self.N_tot = self.Nx * self.Ny * self.N_cell**2
        x=(np.arange(self.Nx*self.N_cell)+1)*self.Nx/(self.Nx*self.N_cell+1)
        y=(np.arange(self.Ny*self.N_cell)+1)*self.Ny/(self.Ny*self.N_cell+1)
        
        self.x, self.y = np.meshgrid(x, y, indexing='ij')
        self.x = np.reshape(self.x, (self.N_tot)) # Flatten into a 1d array
        self.y = np.reshape(self.y, (self.N_tot))
        self.vx = np.zeros(self.x.size) # Initialize zero velocity arrays
        self.vy = np.zeros(self.y.size)
        
        # Determine relative particle weights wf to give intended n_e
        # 1 more layer of ghost cells to allow linear interpolation without 
        # extrapolation
        self.wf = np.pad(self.n_e, (1,1), mode='edge')
        wf_x, wf_y = np.arange(self.Nx +2)-0.5, np.arange(self.Ny +2)-0.5
        self.wf = interpn((wf_x, wf_y), 
                          self.wf,
                          (self.x, self.y),
                          method='linear')
        
        # Remove particles with wf=0 (in case I0 has holes in it, eg a grid)
        self.x=self.x[self.wf>0] 
        self.y=self.y[self.wf>0] 
        self.vx=self.vx[self.wf>0] 
        self.vy=self.vy[self.wf>0] 
        self.wf=self.wf[self.wf>0]
        self.N_tot=self.x.size
        print(f"Total number of particles {self.N_tot:.1e}")
        
        # Store initial position to calculate displacement
        self.x0= np.copy(self.x)
        self.y0= np.copy(self.y)


        # Find the actual n_e on the grid by linear weighting of particles to grid
        # Will be lower in the ghost cells and higher on the inner cells because do
        # not have particles outside the grid and there may be rounding errors
        self._weight_n_e()
        
        # Ensure the mean of the electron density excluding the ghost cells is 1
        norm=np.mean(self.n_e[1:-1, 1:-1])
        self.n_e *= 1/norm
        self.wf *= 1/norm
        
        # Reconstructed I0 (remove ghost cells and undo normalization of initial n_e)
        self.I0_rec=self.n_e[1:-1, 1:-1]*np.mean(self.I0)
        # Save a copy of n_e to normalize the resulting force with
        self.n0 = np.copy(self.n_e)
        
        # Drag coefficient on the grid set to twice the plasma frequency when
        # n_e=n_i to give critical damping at desired final state
        self.drag = 2*np.sqrt(self.n_i)
        
        # Coordinates on which f_d is defined (cell centers)
        self.xc, self.yc = np.arange(self.Nx)+0.5, np.arange(self.Ny)+0.5   
     
        # Create variables for convolution matrices
        # When None, these will cause _calculate_E() to create the 
        # matrices
        self.Cx, self.Cy = None, None
        self._calculate_E()

        
        
        
    def _calculate_E(self):
        """
        Calculate E-field from n_i and n_e

        """
        if self.twoD:
            self._calculate_E_2d()
        else:
            self._calculate_E_1d()
            
    def _calculate_E_1d(self):
        # Electric field from dE/dz=e(n_i-n_e)/epsilon_0 with e=1, epsilon_0=1, dz=1
        # at cell edges (enforce E=0 on boundaries)
        self.Ex = np.cumsum(self.n_i[1:-1] -self.n_e[1:-1])
        
        # Pad E with one cell of zeros all around
        self.Ex = np.pad(self.Ex, (1,1), mode='constant', constant_values=((0,0)) )

            
    def _calculate_E_2d(self):
        """
        Calculates the electric field from n_e and n_i
        
        This is included as a method rather than a separate function because 
        numba does not support scipy functions. 
        
        TODO: write a numba-compatible convolve2d? 
        """
        
        # This block runs the first time this function is called
        if self.Cx is None:
            # Convolution matrices to determine electric field from charge on grid, 
            # given by the electric field of a point charge at the center of a grid 
            # twice the size of the actual grid, not memory efficient but faster than a 
            # parfor loop
            Cx_x,Cx_y = np.meshgrid(np.arange(-self.Nx, self.Nx-1)+0.5,
                                   np.arange(-self.Ny+1, self.Ny),
                                   indexing='ij')     
            Cy_x, Cy_y = np.meshgrid(np.arange(-self.Nx+1, self.Nx),
                                   np.arange(-self.Ny, self.Ny-1)+0.5,
                                   indexing='ij') 
            
            self.Cx=Cx_x/(Cx_x**2+Cx_y**2)/2/np.pi 
            self.Cy=Cy_y/(Cy_x**2+Cy_y**2)/2/np.pi
        
            # Create axes for the grids upon which the Electric fields are 
            # defined
            self.xEx, self.yEx = np.arange(self.Nx+1), np.arange(self.Ny+2)-0.5
            self.xEy, self.yEy = np.arange(self.Nx+2)-0.5, np.arange(self.Ny+1)
        
        self.Ex = convolve2d(self.n_i-self.n_e, self.Cx, mode='full')
        self.Ex = self.Ex[self.Nx-1:2*self.Nx,
                          self.Ny-2:2*self.Ny]
        self.Ey = convolve2d(self.n_i-self.n_e, self.Cy, mode='full')
        self.Ey = self.Ey[self.Nx-2:2*self.Nx,
                          self.Ny-1:2*self.Ny]


    def _interpolate_E(self):
        """
        Find E at particle positions by linear interpolation
         
        If a particle leaves the grid put E=0
        """
        
        if self.twoD:
            self.Ex_p = interpn((self.xEx, self.yEx),
                                self.Ex,
                                (self.x, self.y),
                                 method='linear', bounds_error=False,
                                 fill_value = 0.0)
            
            self.Ey_p = interpn((self.xEy, self.yEy),
                                self.Ey,
                                (self.x, self.y),
                                 method='linear', bounds_error=False,
                                 fill_value = 0.0)
        else:
           
            self.Ex_p = interpn((self.x_grid,), self.Ex, (self.x,),
                                method='linear',
                                bounds_error=False,
                                fill_value = 0.0)
            
    def _calculate_dt(self):
        if self.twoD:
                self.dt=1/np.max(1+np.sqrt(self.vx**2 + self.vy**2) +
                            np.sqrt(self.Ex_p**2 + self.Ey_p**2) )
        else:
                self.dt=1/np.max(1+np.abs(self.vx)+np.abs(self.Ex_p))
                
                
    def _energy(self):
        """
        Calculates the current energy of the simulation. 
        
        At the beginning, vx=vy=0 so this is just the electric field energy,
        and that value alone is used to determine ``energy_stop``.
        """
        
        if self.twoD:
            energy=(np.sum(self.Ex**2) + np.sum(self.Ey**2) + 
                    np.sum(self.wf*(self.vx**2+self.vy**2)) )
        else:
            energy=np.sum(self.Ex**2)+ np.sum(self.wf*self.vx**2)
            
        return energy
    

            
    def _interpolate_drag(self):
        """
        Interpolate the drag coefficient at each particle position
        """
        if self.twoD:
            drag_p = interpn((self.xc, self.yc), 
                                   self.drag, 
                                   (self.x, self.y),
                                   method='nearest',
                                   bounds_error=False,
                                   fill_value=2.0)

        else:
            drag_p = interpn((self.x_grid,), self.drag, (self.x,),
                                  method='nearest',
                                   bounds_error=False,
                                   fill_value=2.0)
            
        
        self.drag_p = np.exp(-drag_p*self.dt)
        
        
        
    def _weight_n_e(self):
        """
        Calculate a new n_e by weighting particles back onto the grid.
        """
        # Find the actual n_e on the grid by linear weighting of particles to grid
        # Will be lower in the ghost cells and higher on the inner cells because do
        # not have particles outside the grid and there may be rounding errors
        
        
        if self.twoD:
            self.n_e = _weight_2d(self.x, self.y, self.wf, 
                                      self.Nx, self.Ny)
        else:
            self.n_e = _weight_1d(self.x, self.wf, self.Nx)



    def _plot_status(self):
        
        if self.twoD:
            fig, axarr = plt.subplots(nrows=3, figsize=(3,9))
            ax=axarr[0]
            ax.set_aspect('equal')
            ax.set_title(f"n_e, Iter: {self.n_iter+1}")
            ax.pcolormesh(self.n_e.T*np.mean(self.I))
            
            ax=axarr[1]
            ax.set_title("Ex")
            ax.set_aspect('equal')
            ax.pcolormesh(self.Ex.T, cmap='seismic')
            
            ax=axarr[2]
            ax.set_title("Ey")
            ax.set_aspect('equal')
            ax.pcolormesh(self.Ey.T, cmap='seismic')
            plt.show()
            
        else:
            fig, ax = plt.subplots()
            ax.plot(self.n_e*np.mean(self.I), label='n_e')
            ax2 = ax.twinx()
            ax2.plot([], [], label='n_e')
            ax2.plot(self.Ex, label='Ex')
            ax2.legend(loc='upper right')

    def run(self):
        # Energy criterion for convergence (use 2*energy)
        self.energy_stop = self.tolerance * self._energy()

        # Main loop
        iterate=True
        self.n_iter=0 # n_iter number of iterations
        self.energies = []
        
        print("Iterating")
        while iterate:
            print(f"Iteration: {self.n_iter+1}")
            
            if self.plots:
                self._plot_status()
            
            self._interpolate_E()
            
            self._calculate_dt()
            
            self._interpolate_drag()
            
            # Update particle velocities with drag and electric field
            # and the particle positions
            self.vx=self.vx*self.drag_p-self.Ex_p*self.dt
            self.x= self.x + self.vx*self.dt
            if self.twoD:
                self.vy=self.vy*self.drag_p-self.Ey_p*self.dt
                self.y= self.y + self.vy*self.dt
                
            # Update electron density and electric field
            self._weight_n_e()

            # Update the E-field
            self._calculate_E()
            

            # Update total energy
            energy = self._energy()
                
            self.energies.append(energy)
                
            if self.verbose:
                print(f"Energy: {energy:.1e} (self.energy_stop: {self.energy_stop:.1e})")
            
            # Determine if energy has fallen sufficiently to stop iterating
            if energy < self.energy_stop :
                iterate=False
                print(f'Energy tolerance reached at iteration {self.n_iter}')
            elif self.n_iter >= self.max_iter:
                iterate=False
                print('Iteration limit reached')
                
                
            self.n_iter += 1
                
                            
        print(f'Final energy fraction {energy/self.energy_stop*self.tolerance:.2f}')
        
        if self.twoD:
            # Weight particle displacements from initial position z0 to the grid 
            # to determine F then remove ghost cells
            self.Fx = _weight_2d(self.x0, self.y0, self.wf*(self.x-self.x0), 
                                 self.Nx, self.Ny)/self.n0
            self.Fx = self.Fx[1:-1, 1:-1]
            
            self.Fy = _weight_2d(self.x0, self.y0, self.wf*(self.y-self.y0), 
                                 self.Nx, self.Ny)/self.n0
            self.Fy = self.Fy[1:-1, 1:-1]
            
            # Find reconstructed measured proton intensity (remove ghost cells and undo
            # the normalization of the final n_e) 
            self.I_rec=self.n_e[1:-1, 1:-1]*np.mean(self.I)
            
        else:
            # Weight particle displacements from initial position z0 to the grid 
            # to determine F then remove ghost cells
            self.F = _weight_1d(self.x0, self.wf*(self.x-self.x0), self.Nx)/self.n0
            self.F = self.F[1:-1]
            
            # Find reconstructed measured proton intensity (remove ghost cells and undo
            # the normalization of the final n_e) 
            self.I_rec=self.n_e[1:-1]*np.mean(self.I)
            
            
    @property
    def results(self):
        if self.twoD:
            return self.I_rec, self.Fx, self.Fy
        else:
            return self.I_rec, self.F

        
        
        
        
    





            
        