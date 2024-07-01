"""
mongeampere.py 
@author: Peter Heuer

This Monge-Ampere solver is translated from a matlab program by Jonathan Davies
which itself is adapted from the PROBLEM code by Archie Bott:

https://github.com/flash-center/PROBLEM

If you use this code, please cite the following papers 

J.R. Davies, P.V. Heuer, A.F.A. Bott Quantitative proton radiography and shadowgraphy for arbitrary intensities.
High Energy Density Physics 49, (2023) https://doi.org/10.1016/j.hedp.2023.101067

A.F.A. Bott et al. Proton imaging of stochastic magnetic fields. Journal of
Plasma Physics 83, (2017) https://doi.org/10.1017/S0022377817000939

"""


import numpy as np
from scipy.interpolate import interpn, griddata


class InvertMA:
    
    def __init__(self, 
                 I:np.ndarray, 
                 I0: np.ndarray=None, 
                 fdt: float = None, 
                 tolerance: float = None, 
                 max_iter: int = None, 
                 plots=False, 
                 verbose=True):
        
        # Cast to 2D array if necessary
        if I.ndim == 1:
            self.I = I[:, np.newaxis]
        elif I.ndim == 2:
            self.I = I
        else:
            raise ValueError("I must be either a 1D or 2D array.")
        
        if I0 is None:
            print("I0 not specified, using a uniform source intensity profile.")
            self.I0 = np.ones(self.I.shape)
        elif np.any(I0<0):
            raise ValueError("All elements of I0 must be >0")
        elif np.any(np.isnan(I0)):
            raise ValueError("I0 cannot contain NaN values.")
        elif I0.ndim == 1:
            self.I0 = I0[:, np.newaxis]
        elif I0.ndim == 2:
            self.I0 = I0
        else:
            raise ValueError("I0 must be either a 1D or 2D array.")
            
            
        if self.I0.shape != self.I.shape:
            raise ValueError(f"I and I0 must have the same shape, but {self.I.shape} != {self.I0.shape}")
        
        
        
        if fdt is None:
            self.fdt = 0.25
            print("Timestep multiplier fdt not specified, using default value 0.25")
        elif fdt < 0 or fdt > 1:
            raise ValueError(f"fdt must be 1 > fdt > 0, but provided value is {fdt}")
        else:
            self.fdt = fdt
            
        if tolerance is None:
            self.tolerance = 1e-3
            print("Tolerance not specified, using default value 1e-3")   
        elif tolerance < 0 or tolerance > 1:
            raise ValueError(f"tolerance must be 1 > tolerance > 0, but provided value is {tolerance}")
        else:
            self.tolerance = tolerance
            
        if max_iter is None:
            self.max_iter = int(1e6)
            print("Maximum iteration max_iter not specified, using default value 1e6")
        elif max_iter < 1: 
            raise ValueError(f"max_iter must be >1, but provided value is {max_iter}")
        else:
            self.max_iter = int(max_iter)
            
        self.plots = plots
        self.verbose = verbose
            
        self._setup()
            
            
            
    def _setup(self):
        # Make sure I and I0 are double precision, add one ghost cell in each direction
        self.I = self.I.astype(np.double)
        self.I = np.pad(self.I, (1,1), mode='edge')
        self.I0 = self.I0.astype(np.double)
        self.I0 = np.pad(self.I0, (1,1), mode='edge')
        
        # Store the mean of I so that I_rec matches the original peaks with
        # any zeros filled in
        self.meanI = np.mean(self.I)
        self.I /= np.mean(self.I)
        self.I0 /= np.mean(self.I0)
        
        # Replace any zeros in I and I0 with the minimum non-zero value
        # in that array
        Imin = np.min(self.I[self.I != 0])
        I0min = np.min(self.I0[self.I0 != 0])
        self.I[self.I==0] = Imin
        self.I0[self.I0==0] = I0min
        
        # Normalize so that mean=1, store meanI for final I_rec
        self.I /= np.mean(self.I)
        self.I0 /= np.mean(self.I0)
        
        self.Nx, self.Ny = self.I.shape
        
        # Create a grid of bin centers
        # Note that changing the origin of x,y made no difference: here it
        # is placed at the center
        # x<->y switched when adapting matlab to python
        self.y, self.x = np.meshgrid(np.arange(-self.Nx/2+0.5, self.Nx/2+0.5),
                                     np.arange(-self.Ny/2+0.5, self.Ny/2+0.5),
                                     indexing='ij')
        
        # Initial value of deflection potential phi+(x^2+y^2)/2
        # Note: uses F=grad(phi), the opposite of the normal sign convention 
        self.phi = (self.x**2 + self.y**2)/2
        
        # Initial values of derivatives of phi+(x^2+y^2)/2 (x,y indicate derivatives)
        self.phix = np.copy(self.x)
        self.phiy = np.copy(self.y)
        self.phixy = np.zeros([self.Nx, self.Ny])
        self.phixx = np.ones([self.Nx, self.Ny])
        self.phiyy = np.copy(self.phixx)
        # Initial value of dphi_dt
        self.dphi_dt=np.log(self.I/self.I0)
        
        
    def run(self):
        
        #Initialize records of time step and dphi_dtMS
        dt=np.zeros(self.max_iter) 
        dphi_dtMS=np.zeros(self.max_iter)
        
        # Some useful indices to shorten expressions for derivatives
        ic=np.arange(2, self.Nx) -1 # -1 because matlab -> python
        ip=ic+1
        im=ic-1
        jc=np.arange(2, self.Ny) -1
        jp=jc+1
        jm=jc-1

        iterate = True
        n_iter = 0
        

        
        print("Iterating...")
        while iterate:
            n_iter += 1
            if n_iter % 1000 == 0 and self.verbose:
                print(f"i={n_iter}")

            # Adaptive timestep to prevent points crossing
            # Find the smallest separation in x and y
            dx=np.min(np.min(np.diff(self.phix,n=1,axis=1)))
            dy=np.min(np.min(np.diff(self.phiy, n=1, axis=0)))
            
            # Find the maximum absolute effective vx and vy
            vx=np.max(np.abs(self.dphi_dt[:,jp]-self.dphi_dt[:,jm]))/2 
            vy=np.max(np.abs(self.dphi_dt[ip,:]-self.dphi_dt[im,:]))/2
         
            
    
            # Combine the limits 1, dx/vx and dy/vy applying a multiplier fdt
            dt[n_iter-1]=self.fdt/(1+vx/dx+vy/dy)
            
            # Update phi+(x^2+y^2)/2
            self.phi += + dt[n_iter-1]*self.dphi_dt

            # Update derivatives of phi+(x^2+y^2)/2
            # BCs: phi has zero perpendicular gradients so phix=x and phiy=y at
            # respective boundaries because we are adding (x^2+y^2)/2, phixy is zero
            # at boundaries, only phixx and phiyy boundary values need to be updated
            
            self.phix[:,jc]=(self.phi[:,jp]-self.phi[:,jm])/2
            self.phiy[ic,:]=(self.phi[ip,:]-self.phi[im,:])/2 
            self.phixy[ic,jc]=(self.phi[ip,jp]-self.phi[im,jp]-self.phi[ip,jm]+self.phi[im,jm])/4
            
        
            
            self.phixx[:, jc]=self.phi[:, jp]-2*self.phi[:, jc]+self.phi[:, jm]
            self.phixx[:, 0]=2*(self.phi[:, 1]-self.phi[:, 0]-self.x[:, 0])
            self.phixx[:, self.Ny-1]=2*(self.phi[:, self.Ny-2]-self.phi[:, self.Ny-1]+self.x[:, self.Ny-1])
            
            self.phiyy[ic,:]=self.phi[ip,:]-2*self.phi[ic,:]+self.phi[im,:]
            self.phiyy[0,:]=2*(self.phi[1,:]-self.phi[0,:]-self.y[0,:])
            self.phiyy[self.Nx-1,:]=2*(self.phi[self.Nx-2,:]-self.phi[self.Nx-1,:]+self.y[self.Nx-1,:])
            
            # Update I_rec using the determinant of the Jacobian of the positions
            # phix and phiy (I_0(x,y)dxdy to I=I_0(phix,phiy)dphixdphiy)
            self.I_rec=self.I0/np.abs(self.phixx*self.phiyy-self.phixy**2)
            
            if np.any(~np.isfinite(self.I_rec)):
                #print(self.I_rec)
                raise ValueError('Routine has gone unstable, try a smaller fdt')
                
                
            # Some flat arrays for interpolation
            x_flat = self.y[:, 0]
            y_flat = self.x[0, :]
            phix_flat = self.phiy.flatten()
            phiy_flat = self.phix.flatten()
            
            # Update dphi_dt
            # linear interpolation of I to positions phix,phiy
            # tried nearest and it made little difference
            interp_I = interpn((x_flat, y_flat),
                               self.I,
                               (phix_flat, phiy_flat),
                               method='linear')
            interp_I = np.reshape(interp_I, self.I_rec.shape)
            

            self.dphi_dt=np.log(interp_I/self.I_rec)
            # Update mean squared dphi_dt
            dphi_dtMS[n_iter-1]=np.mean(self.dphi_dt[:]**2)
            
            

            # Determine if dphi_dt is small enough to stop iterating
            if dphi_dtMS[n_iter-1] < self.tolerance:
                iterate=False
                print(f"Tolerance reached at iteration {n_iter}")
       
            if n_iter == self.max_iter:
                iterate=False
                print("Iteration limit reached")
                
  
        # Find line-integrated forces, dropping the ghost cells
        self.Fy=self.phix[1:self.Nx-1, 1:self.Ny-1]-self.x[1:self.Nx-1, 1:self.Ny-1]
        self.Fx=self.phiy[1:self.Nx-1, 1:self.Ny-1]-self.y[1:self.Nx-1, 1:self.Ny-1]
        

        
        # Find reconstructed measured intensity at original bin centers
        # linear interpolation for irregularly spaced data
        self.I_rec=self.meanI*griddata((self.phix.flatten(),self.phiy.flatten()),
                                       self.I_rec.flatten(), 
                                       (self.x[1:self.Nx-1, 1:self.Ny-1],
                                        self.y[1:self.Nx-1, 1:self.Ny-1]))

        
        # Retain only valid values of dphi_dtMS and dt
        dphi_dtMS=dphi_dtMS[:n_iter]
        dt=dt[:n_iter]
        
        self.results = (self.I_rec, self.Fx, self.Fy, dphi_dtMS, dt)
        
