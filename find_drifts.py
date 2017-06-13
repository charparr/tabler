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
from skimage import io
from scipy.ndimage import gaussian_filter
import scipy.ndimage
from skimage.measure import regionprops
from skimage.measure import label
from skimage.measure import find_contours
from skimage.morphology import remove_small_objects
from skimage.morphology import remove_small_holes
import pandas as pd



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

years = ['2012','2013','2015','2016']
 
# 11360 x 2340 
bare_surf = Surface('/home/cparr/surfaces/level_1_surfaces/hv/bare_earth/hv_2012_158_bare_earth_dem.tif')
slope_surf = Surface('/home/cparr/surfaces/slope_surfaces/hv_slope.tif')
curve_surf = Surface('/home/cparr/surfaces/slope_surfaces/hv_curve.tif')

snow_depth_surf16 = Surface('/home/cparr/surfaces/depth_ddems/hv/hv_2016_096_depth.tif')
snow_depth_surf15 = Surface('/home/cparr/surfaces/depth_ddems/hv/hv_2015_096_depth.tif')
snow_depth_surf13 = Surface('/home/cparr/surfaces/depth_ddems/hv/hv_2013_103_depth.tif')

#11358 x 2340
snow_depth_surf12 = Surface('/home/cparr/surfaces/depth_ddems/hv/hv_2012_107_depth.tif')

# 11356 x 2340
slope_w12 = Surface('/home/cparr/surfaces/slope_surfaces/hv_w12_slope_clip.tif')
# 11358 x 2340
slope_w13 = Surface('/home/cparr/surfaces/slope_surfaces/hv_w13_slope_clip.tif')
# 11358 x 2340
slope_w15 = Surface('/home/cparr/surfaces/slope_surfaces/hv_w15_slope_clip.tif')
# 11360 x 2340
slope_w16 = Surface('/home/cparr/surfaces/slope_surfaces/hv_w16_slope_clip.tif')

all_surfs = [bare_surf,slope_surf,curve_surf,snow_depth_surf12,
             snow_depth_surf13,snow_depth_surf15,snow_depth_surf16,slope_w12,
             slope_w13,slope_w15,slope_w16]

for s in all_surfs:
    
    diff = s.arr.shape[0]-11356
    s.arr = s.arr[diff:]
    print (s.arr.shape)

mean_wslope = (slope_w12.arr+slope_w13.arr+slope_w15.arr+slope_w16.arr)/4

mean_depth = (snow_depth_surf12.arr+snow_depth_surf13.arr+
                        snow_depth_surf15.arr+snow_depth_surf16.arr)/4
              
# find _drifts
step1 = gaussian_filter(mean_depth,sigma=2)
step2 = (step1 > np.nanmean(step1)+np.nanstd(step1))
step3  = remove_small_objects(step2)
step4  = remove_small_holes(step3)
step5 = step4 * 1
drifts = step4 * mean_depth


contours = find_contours(step5,0.5)

lab_img = label(step4)

snow_props = regionprops(lab_img,mean_depth)
wslope_props = regionprops(lab_img, mean_wslope)
slope_props = regionprops(lab_img, slope_surf.arr)

i=0
idx = []
for r in snow_props:
    r.name = 'drift_' + str(i)
    idx.append(r.name)
    i += 1

i=0
for r in slope_props:
    r.name = 'drift_' + str(i)
    i += 1
    
i=0
for r in wslope_props:
    r.name = 'drift_' + str(i)
    i += 1

# Dump Results to DataFrame

snowfields = ['area', 'bbox', 'centroid', 'convex_area',
       'convex_image', 'coords', 'eccentricity',
       'equivalent_diameter', 'euler_number', 'extent',
       'image', 'label',
       'major_axis_length', 'max_intensity', 'mean_intensity',
       'min_intensity', 'minor_axis_length',
       'moments_hu', 'orientation', 'perimeter',
       'solidity', 'weighted_centroid','weighted_moments_hu']

slopefields=['slope max_intensity', 'slope mean_intensity',
             'slope min_intensity','slope weighted_centroid',
             'slope weighted_moments_hu']

wslopefields=['wslope max_intensity', 'wslope mean_intensity',
             'wslope min_intensity','wslope weighted_centroid',
             'wslope weighted_moments_hu']

df = pd.DataFrame(index=idx, columns = snowfields+slopefields+wslopefields)


for r in snow_props:
    for c in snowfields:
            df.ix[r.name][c]=r[c]
for r in slope_props:
    for c in slopefields:
        n = c[6:]
        df.ix[r.name][c]=r[n]
for r in wslope_props:
    for c in wslopefields:
        n = c[7:]
        df.ix[r.name][c]=r[n]

for j in range(7):
    df['moments_hu'+str(j)]=[i[j] for i in df['moments_hu']]
    df['weighted_moments_hu'+str(j)]=[i[j] for i in df['weighted_moments_hu']]
    df['slope weighted_moments_hu'+str(j)]=[i[j] for i in df['slope weighted_moments_hu']]
    df['wslope weighted_moments_hu'+str(j)]=[i[j] for i in df['wslope weighted_moments_hu']]

del df['moments_hu'],df['weighted_moments_hu'],df['slope weighted_moments_hu'],df['wslope weighted_moments_hu']
df.columns





#bigdf = df[df['area']>500]
#
fig, ax = plt.subplots()
ax.imshow(mean_depth, interpolation='nearest', cmap='gray',vmin=0,vmax=2)

for n, reg in enumerate(snow_props):
    ax.plot(reg.coords[:, 1], reg.coords[:, 0], linewidth=1,alpha=0.5,c='m')
    #ax.text(reg.centroid[1], reg.centroid[0], reg.name)
plt.show()
    


