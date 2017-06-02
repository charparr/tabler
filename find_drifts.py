#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 14:42:07 2017

@author: cparr

This script will try to identify drift starts based on terrain parameters like
slope and curvature. Once points are identified, we can back away some 45 m in
a variety of directions. We can also check the snow data for regions of 
continuous depth above some threshold, i.e. one sigma above the mean.
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
from math import factorial
from skimage import io
from skimage.measure import profile_line
from scipy.ndimage import gaussian_filter
from scipy import integrate
import scipy.ndimage
import matplotlib.patches as patches
from astropy.convolution import convolve_fft
# Global Methods



class Surface(object):
    '''A class for bare earth and snow surfaces.
    Each instance is initialized to mask out common NoData values and remove
    empty rows. The surface is also resampled to 1 by 1 meter data to make
    profile and flux calculations more straightforward.
    '''
    
    def __init__(self, fpath):
        self.arr = io.imread(fpath)
        self.arr[(self.arr < -10)] = np.nan
        rowmask = np.all(np.isnan(self.arr), axis=1)
        self.arr = self.arr[~rowmask]
        self.arr = scipy.ndimage.zoom(self.arr, 2, order = 1)
        
    def subset_surf(self, n_divs, names):
        '''Subset the surface into smaller chunks. This speeds up analysis and
        makes plotting a bit easier. Also removes empty columns. Subsets are 
        stored in a dict with user defined keys.
        Parameters
        ----------
        n_divs : int
            the number of vertical subsets.
        names: list of strings for keys
        '''
        ysize = round(self.arr.shape[0] / n_divs)
        
        self.subdict = dict.fromkeys(names)
        
        for nam, num in zip(names, range(0, n_divs)):
            
            self.subdict[nam] = self.arr[(ysize * num):(ysize * (num+1))]
            colmask = np.nansum(self.subdict[nam],axis=0)==0
            emptycols = np.where(colmask)
            nd_boundaries = [j-i for i, j in zip(emptycols[0][:-1], emptycols[0][1:])]
        
            if max(nd_boundaries, default='no elements') == 1:
                if emptycols[0][0]==0:
                    self.subdict[nam] = self.subdict[nam][::, emptycols[0].max():]
                else:
                    self.subdict[nam] = self.subdict[nam][::, :emptycols[0].min()]
            elif max(nd_boundaries, default='no elements') == 'no elements':
                pass
            else:
                l_edge = np.where(np.array(nd_boundaries) > 1)[0][0]
                r_edge = l_edge + max(nd_boundaries)
                self.subdict[nam] = self.subdict[nam][::,l_edge:r_edge]
                
#class Surface_Derivs(object):
#
#    def __init__(self,surf):
#        self.surf = surf
#        
#        y = self.surf.copy()
#        window_size = 5
#        order = 1
#        rate = 1
#        deriv = 1
#        
#        order_range = range(order+1)
#        half_window = (window_size -1) // 2
#        # precompute coefficients
#        b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
#        m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
#        # pad the signal at the extremes with
#        # values taken from the signal itself
#        firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
#        lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
#        y = np.concatenate((firstvals, y, lastvals))
#        self.slope =  np.convolve( m[::-1], y, mode='valid')
        
def sgolay2d ( z, window_size, order, derivative=None):
    """
    """
    # number of terms in the polynomial expression
    n_terms = ( order + 1 ) * ( order + 2)  / 2.0

    if  window_size % 2 == 0:
        raise ValueError('window_size must be odd')

    if window_size**2 < n_terms:
        raise ValueError('order is too high for the window size')

    half_size = window_size // 2

    # exponents of the polynomial. 
    # p(x,y) = a0 + a1*x + a2*y + a3*x^2 + a4*y^2 + a5*x*y + ... 
    # this line gives a list of two item tuple. Each tuple contains 
    # the exponents of the k-th term. First element of tuple is for x
    # second element for y.
    # Ex. exps = [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2), ...]
    exps = [ (k-n, n) for k in range(order+1) for n in range(k+1) ]

    # coordinates of points
    ind = np.arange(-half_size, half_size+1, dtype=np.float64)
    dx = np.repeat( ind, window_size )
    dy = np.tile( ind, [window_size, 1]).reshape(window_size**2, )

    # build matrix of system of equation
    A = np.empty( (window_size**2, len(exps)) )
    for i, exp in enumerate( exps ):
        A[:,i] = (dx**exp[0]) * (dy**exp[1])

    # pad input array with appropriate values at the four borders
    new_shape = z.shape[0] + 2*half_size, z.shape[1] + 2*half_size
    Z = np.zeros( (new_shape) )
    # top band
    band = z[0, :]
    Z[:half_size, half_size:-half_size] =  band -  np.abs( np.flipud( z[1:half_size+1, :] ) - band )
    # bottom band
    band = z[-1, :]
    Z[-half_size:, half_size:-half_size] = band  + np.abs( np.flipud( z[-half_size-1:-1, :] )  -band )
    # left band
    band = np.tile( z[:,0].reshape(-1,1), [1,half_size])
    Z[half_size:-half_size, :half_size] = band - np.abs( np.fliplr( z[:, 1:half_size+1] ) - band )
    # right band
    band = np.tile( z[:,-1].reshape(-1,1), [1,half_size] )
    Z[half_size:-half_size, -half_size:] =  band + np.abs( np.fliplr( z[:, -half_size-1:-1] ) - band )
    # central band
    Z[half_size:-half_size, half_size:-half_size] = z

    # top left corner
    band = z[0,0]
    Z[:half_size,:half_size] = band - np.abs( np.flipud(np.fliplr(z[1:half_size+1,1:half_size+1]) ) - band )
    # bottom right corner
    band = z[-1,-1]
    Z[-half_size:,-half_size:] = band + np.abs( np.flipud(np.fliplr(z[-half_size-1:-1,-half_size-1:-1]) ) - band )

    # top right corner
    band = Z[half_size,-half_size:]
    Z[:half_size,-half_size:] = band - np.abs( np.flipud(Z[half_size+1:2*half_size+1,-half_size:]) - band )
    # bottom left corner
    band = Z[-half_size:,half_size].reshape(-1,1)
    Z[-half_size:,:half_size] = band - np.abs( np.fliplr(Z[-half_size:, half_size+1:2*half_size+1]) - band )

    # solve system and convolve
    if derivative == None:
        m = np.linalg.pinv(A)[0].reshape((window_size, -1))
        return convolve_fft(Z, m, interpolate_nan=True)
    elif derivative == 'col':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        return convolve_fft(Z, -c)
    elif derivative == 'row':
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return convolve_fft(Z, -r)
    elif derivative == 'both':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return convolve_fft(Z, -r), convolve_fft(Z,-c)
    
#    def savitzky_golay(self, y, window_size, order, deriv=0, rate=1):
#         """Smooth (and optionally differentiate) with a Savitzky-Golay filter.
#         The filter removes high frequency noise from data.
#         It has the advantage of preserving the original shape and
#         features of the signal better than other types of filtering
#         approaches, such as moving averages techniques.
#         Parameters
#         ----------
#         y : array_like, shape (N,)
#             the values of the time history of the signal.
#         window_size : int
#             the length of the window. Must be an odd integer number.
#         order : int
#             the order of the polynomial used in the filtering.
#             Must be less then `window_size` - 1.
#         deriv: int
#             order of the derivative to compute (default = 0 is only smoothing)
#         Returns
#         -------
#         ys : ndarray, shape (N)
#             the smoothed signal (or it's n-th derivative).
#         Notes
#         -----
#         The Savitzky-Golay is a type of low-pass filter, particularly
#         suited for smoothing noisy data. The main idea behind this
#         approach is to make for each point a least-square fit with a
#         polynomial of high order over a odd-sized window centered at
#         the point.
#         Examples
#         --------
#         t = np.linspace(-4, 4, 500)
#         y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
#         ysg = savitzky_golay(y, window_size=31, order=4)
#         import matplotlib.pyplot as plt
#         plt.plot(t, y, label='Noisy signal')
#         plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
#         plt.plot(t, ysg, 'r', label='Filtered signal')
#         plt.legend()
#         plt.show()
#         References
#         ----------
#         .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
#            Data by Simplified Least Squares Procedures. Analytical
#            Chemistry, 1964, 36 (8), pp 1627-1639.
#         .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
#            W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
#            Cambridge University Press ISBN-13: 9780521880688
#         """
#     
#         try:
#             window_size = np.abs(np.int(window_size))
#             order = np.abs(np.int(order))
#         except:
#             raise ValueError("window_size and order have to be of type int")
#         if window_size % 2 != 1 or window_size < 1:
#             raise TypeError("window_size size must be a positive odd number")
#         if window_size < order + 2:
#             raise TypeError("window_size is too small for the polynomials order")
#         order_range = range(order+1)
#         half_window = (window_size -1) // 2
#         # precompute coefficients
#         b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
#         m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
#         # pad the signal at the extremes with
#         # values taken from the signal itself
#         firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
#         lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
#         y = np.concatenate((firstvals, y, lastvals))
#         return np.convolve( m[::-1], y, mode='valid')    

years = ['2012','2013','2015','2016']
  
bare_surf = Surface('/home/cparr/surfaces/level_1_surfaces/hv/bare_earth/hv_2012_158_bare_earth_dem.tif')
snow_depth_surf16 = Surface('/home/cparr/surfaces/depth_ddems/hv/hv_2016_096_depth.tif')
snow_depth_surf15 = Surface('/home/cparr/surfaces/depth_ddems/hv/hv_2015_096_depth.tif')
snow_depth_surf13 = Surface('/home/cparr/surfaces/depth_ddems/hv/hv_2013_103_depth.tif')
snow_depth_surf12 = Surface('/home/cparr/surfaces/depth_ddems/hv/hv_2012_107_depth.tif')
                
bare_surf.subset_surf(12, ['b1','b2','b3','b4',
                           'b5','b6','b7','b8',
                           'b9','b10','b11','b12'])
    
snow_depth_surf16.subset_surf(12, ['2016_1','2016_2','2016_3','2016_4',
                                 '2016_5','2016_6','2016_7','2016_8',
                                 '2016_9','2016_10','2016_11','2016_12'])
    
snow_depth_surf15.subset_surf(12, ['2015_1','2015_2','2015_3','2015_4',
                                 '2015_5','2015_6','2015_7','2015_8',
                                 '2015_9','2015_10','2015_11','2015_12'])
    
snow_depth_surf13.subset_surf(12, ['2013_1','2013_2','2013_3','2013_4',
                                 '2013_5','2013_6','2013_7','2013_8',
                                 '2013_9','2013_10','2013_11','2013_12'])
    
snow_depth_surf12.subset_surf(12, ['2012_1','2012_2','2012_3','2012_4',
                                 '2012_5','2012_6','2012_7','2012_8',
                                 '2012_9','2012_10','2012_11','2012_12'])