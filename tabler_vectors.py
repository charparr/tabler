# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import warnings
#from math import factorial
from skimage import io
from skimage.measure import profile_line
#from scipy.ndimage import gaussian_filter
from scipy import integrate
import scipy.ndimage
import matplotlib.patches as patches
import seaborn as sns

class Surface(object):
    '''A class for bare earth and snow surfaces.
    Each instance is initialized to mask out common NoData values and remove
    empty rows. The surface is also resampled to 1 by 1 meter data to make
    profile and flux calculations more straightforward.
    '''
    
    def __init__(self, fpath):
        self.arr = io.imread(fpath)
        self.arr[self.arr == -9999] = np.nan
        self.arr[(self.arr < -10)] = np.nan
        rowmask = np.all(np.isnan(self.arr), axis=1)
        self.arr = self.arr[~rowmask]
        self.arr = scipy.ndimage.zoom(self.arr, 2, order = 1)
        
    def subset_surf(self, n_divs, names):
        '''Subset the surface remove empty columns. Subsets are stored in a 
        dict with user defined keys.
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
        
class Transect(object):
    '''A class for transects along which we can measure flux and generate
    Tabler profiles. We initialize a Transect instance by providing the bare
    earth snow depth surfaces and the beginning and end indices for
    the profile we want. We can initialize Tabler assuming the user is choosing
    the start and end of the drift as the profile.
        Parameters
        ----------
        bare_earth : array_like, shape (N,)
            elevation values of snow free surface
        years : list[str]
            years for which snow depth surfaces are available
        snow_depths : list[arr]
            snow depth surfaces
        angle_deg: int
            angle in degrees TN from which we take the transect (i.e. the
            the wind direction)
        x1, y1: int
            start indices of transect
        length: int
            length of transect in meters
    '''
    
    def __init__(self, bare_earth, years, snow_depths, angle_deg, x1, y1, length):
        
        self.bare_earth = bare_earth # bare earth surface
        self.angle_deg = angle_deg # where does the wind come from?
        self.x1 = x1
        self.y1 = y1
        
        # Compute end of transect
        cosa = np.cos((np.deg2rad(90 + angle_deg)))
        cosb = np.cos((np.deg2rad(angle_deg)))
        self.x2, self.y2 = (x1 +(length * cosa), y1+(length * cosb))
            
        # Construct DEM profile with a width of 1 m.
        self.dem_profile = profile_line(bare_earth,
                                               ((y1,x1)),((self.y2,self.x2)),1)
        
        # Check DEM profile for No Data values and where they occur.
        nan_idx = np.argwhere(np.isnan(self.dem_profile))
        
        # Truncate DEM Profile at the first No Data instance
        if len(nan_idx) > 0:
            
            self.dem_profile = self.dem_profile[:nan_idx[0]-1]
        
        # The DEM profile is a template for the snow depth profile
        
        dem_pro_len = len(self.dem_profile)

        self.x3, self.y3 = (x1 +(dem_pro_len * cosa), y1+(dem_pro_len * cosb))
       
        self.snowdict = dict.fromkeys(years)
        
        # For each year generate profiles and store in dictionary
        for yr, depth in zip(years, snow_depths):
            
            self.snowdict[yr] = {}
            
            self.snowdict[yr]['snow depth surface'] = depth
            self.snowdict[yr]['winter surface'] = depth + self.bare_earth
            
            self.snowdict[yr]['winter surface profile'] = \
                         profile_line(self.snowdict[yr]['winter surface'],
                                      ((y1,x1)),((self.y3,self.x3)),1)[:-1]
                         
            self.snowdict[yr]['depth profile'] = profile_line(depth,((y1,x1)),
                         ((self.y3,self.x3)),1)[:-1]
        
    def TablerProfile(self):
        
        '''
        Here we generate a Tabler Equilibirum Profile for the drift.
        Parameters include the lip of the trap and the drift end, because the
        we need to have more data (e.g. upwind) in the transect. Basically the
        user decides where the drift starts and ends. I have ways to automate
        this but they are not fully tested.
        '''
        # Dynamic Creation of Tabler Profile
        # All Coeffcients march downwind 1 m at a time
        # x2 is now the slope from the snow surface to the ground at a
        # horizontal distance of 15 m downwind.
        
        self.dynamic_tabler = self.dem_profile.copy()
        
        i = 45
        while i < len(self.dem_profile) - 46:
            
            upwind_0_45 = self.dem_profile[i-45] - self.dem_profile[i]
                   
            snow_to_ground = self.dynamic_tabler[i] - self.dem_profile[i+5]
                   
            downwind_15_30 = self.dem_profile[i+15] - self.dem_profile[i+30]
                   
            downwind_30_45 = self.dem_profile[i+30] - self.dem_profile[i+45]
            
            if upwind_0_45 > 0:
                x1 = (upwind_0_45 / 45) * -100
            else:
                x1 = (upwind_0_45 / 45) * 100
                     
            if snow_to_ground > 0:
                x2 = snow_to_ground * -100
            else:
                x2 = snow_to_ground * 100                    
                    
            if downwind_15_30 > 0:
                x3 = (downwind_15_30 / 15) * -100
            else:
                x3 = (downwind_15_30 / 15) * 100
                    
            if downwind_30_45 > 0:
                x4 = (downwind_30_45 / 15) * -100
            else:
                x4 = (downwind_30_45 / 15) * 100
                     
            y = (0.25 * x1) + (0.55 * x2) + (0.15 * x3) + (0.05 * x4)
            
            rise = y / 100
            
            self.dynamic_tabler[i+1] = self.dem_profile[i] - rise
                               
            i+=1
            
    def Compute_Flux(self):
         ''' Use Simpson's rule integration to calculate flux over the drift
         fetch. Values will be m^3 per lineal m, so essentialy we have m^2 which
         makes sense because we are getting the area under the snow depth curve.
         '''
         
         tabler_snow_depth = self.dynamic_tabler - self.dem_profile
         self.tabler_flux = integrate.simps(tabler_snow_depth[46:-46])                 
                                                
         self.mean_flux = 0
         for k in self.snowdict:
             
            snow_depth = self.snowdict[k]['depth profile']
            self.snowdict[k]['flux'] = integrate.simps(snow_depth[46:-46])
            self.mean_flux = self.mean_flux + self.snowdict[k]['flux']
        
         self.mean_flux = self.mean_flux / 4.0
         self.tabler_flux_err = self.mean_flux - self.tabler_flux
         self.flux_err_ratio = self.tabler_flux_err / self.tabler_flux

            
                                                                            
    def PlotMaps(self):
        
        fig = plt.figure()
        ax1=plt.subplot(2,2,1)
        im1=plt.imshow(self.snowdict['2016']['snow depth surface'],
                       vmin = 0, vmax = 2)
        ax1.plot([self.x1, self.x3], [self.y1, self.y3], c='r',
                 linestyle='-', linewidth=2)
        plt.title('2016')
        ax1.set_ylabel('m',size = 7)

        
        ax2=plt.subplot(2,2,2,sharex=ax1)
        plt.imshow(self.snowdict['2015']['snow depth surface'],
                   vmin = 0, vmax = 2)
        ax2.plot([self.x1, self.x3], [self.y1, self.y3], c='r',
                 linestyle='-', linewidth=2)
        plt.title('2015')
        ax2.set_yticks([])
        ax2.set_xticks([])
        
        ax3=plt.subplot(2,2,3,sharey=ax1)
        plt.imshow(self.snowdict['2013']['snow depth surface'],
                   vmin = 0, vmax = 2)
        ax3.plot([self.x1, self.x3], [self.y1, self.y3], c='r',
                 linestyle='-', linewidth=2)
        plt.title('2013')
        ax3.set_xlabel('m',size = 7)

        ax4=plt.subplot(2,2,4, sharex = ax3)
        plt.imshow(self.snowdict['2012']['snow depth surface'],
                   vmin = 0, vmax = 2)
        ax4.plot([self.x1, self.x3], [self.y1, self.y3], c='r',
                 linestyle='-', linewidth=2)
        plt.title('2012')
        ax4.set_yticks([])
        ax4.set_xlabel('m',size = 7)
        
        cbar_ax = fig.add_axes([0.90, 0.15, 0.05, 0.7])
        fig.colorbar(im1, cax=cbar_ax)
        plt.suptitle('Snow Depth [m]')
        plt.subplots_adjust(wspace=0.0,hspace=0.2)
        
        #plt.savefig('/home/cparr/depthmaps.png',dpi=300)

                
    def PlotDepthProfiles(self):
    
        fig, ax = plt.subplots()
        plt.plot(self.snowdict['2012']['depth profile'], ':b1', label = '2012')
        plt.plot(self.snowdict['2013']['depth profile'], ':r2', label = '2013')
        plt.plot(self.snowdict['2015']['depth profile'], ':g3', label = '2015')
        plt.plot(self.snowdict['2016']['depth profile'], ':k4', label = '2016')
        plt.ylabel('Snow Depth [m]')
        plt.xlabel('m')
        plt.legend()
        #plt.savefig('/home/cparr/snowdepth_profiles.png',dpi=300)

        
    def PlotSurfaceProfiles(self):
    
        fig, ax = plt.subplots()
        plt.plot(self.dem_profile, label = 'Bare Earth',c='saddlebrown')
        plt.plot(self.snowdict['2012']['winter surface profile'], '--b',alpha=0.5, label = '2012')
        plt.plot(self.snowdict['2013']['winter surface profile'], '--r',alpha=0.5, label = '2013')
        plt.plot(self.snowdict['2015']['winter surface profile'], '--g',alpha=0.5, label = '2015')
        plt.plot(self.snowdict['2016']['winter surface profile'], '--m',alpha=0.5, label = '2016')
        plt.plot(self.dynamic_tabler, label = 'Dynamic Tabler Surface',c='k')
        plt.xlabel('m')
        plt.ylabel('m')                
        plt.legend()

    def PlotSnowSurfacesOnly(self):
    
        fig, ax = plt.subplots()
        plt.plot(self.snowdict['2012']['winter surface profile'] - self.dem_profile, '--b',alpha=0.5, label = '2012')
        plt.plot(self.snowdict['2013']['winter surface profile'] - self.dem_profile, '--r',alpha=0.5, label = '2013')
        plt.plot(self.snowdict['2015']['winter surface profile'] - self.dem_profile, '--g',alpha=0.5, label = '2015')
        plt.plot(self.snowdict['2016']['winter surface profile'] - self.dem_profile, '--m',alpha=0.5, label = '2016')
        plt.plot(self.dynamic_tabler - self.dem_profile, label = 'Dynamic Tabler Surface',c='k')
        plt.xlabel('m')
        plt.ylabel('m')
        plt.title('Snow Surface - Bare Earth Surface')                
        plt.legend()

    def PlotFlux(self):
        
        fig, ax = plt.subplots()
        plt.suptitle('Integrated Flux [m^3 per lineal m]')
        y = [self.snowdict['2012']['flux'], 
                self.snowdict['2013']['flux'],
                self.snowdict['2015']['flux'],
                self.snowdict['2016']['flux'],
                self.tabler_flux]
        ax.bar(np.arange(5),y)
        ax.set_xticklabels(('','2012','2013','2015','2016','Tabler'))
        ax.set_xlabel('Profile')
        ax.set_ylabel('Flux')


    def PlotAll(self):
        self.PlotMaps()
        self.PlotDepthProfiles()
        self.PlotSurfaceProfiles()
        self.PlotSnowSurfacesOnly()
        self.PlotFlux()

def make_transect_obj(bare, years, snows, angle, x1,y1,length):
    t = Transect(bare, years, snows, angle, x1,y1,length)
    t.TablerProfile()
    t.Compute_Flux()
    return t
        
### globally bring in surfaces and test transects

years = ['2012','2013','2015','2016']
  
bare_surf = Surface('/home/cparr/surfaces/level_1_surfaces/hv/bare_earth/hv_2012_158_bare_earth_dem.tif')
snow_depth_surf16 = Surface('/home/cparr/surfaces/depth_ddems/hv/hv_2016_096_depth.tif')
snow_depth_surf15 = Surface('/home/cparr/surfaces/depth_ddems/hv/hv_2015_096_depth.tif')
snow_depth_surf13 = Surface('/home/cparr/surfaces/depth_ddems/hv/hv_2013_103_depth.tif')
snow_depth_surf12 = Surface('/home/cparr/surfaces/depth_ddems/hv/hv_2012_107_depth.tif')

snow_depth_surf16.arr = snow_depth_surf16.arr[2:]
snow_depth_surf15.arr = snow_depth_surf15.arr[2:]
snow_depth_surf13.arr = snow_depth_surf13.arr[2:]
bare_surf.arr = bare_surf.arr[2:]

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
            

#bare_surf.subset_surf(4, ['b1','b2','b3','b4'])
#    
#snow_depth_surf16.subset_surf(4, ['2016_1','2016_2','2016_3','2016_4'])
#    
#snow_depth_surf15.subset_surf(4, ['2015_1','2015_2','2015_3','2015_4'])
#    
#snow_depth_surf13.subset_surf(4, ['2013_1','2013_2','2013_3','2013_4'])
#    
#snow_depth_surf12.subset_surf(4, ['2012_1','2012_2','2012_3','2012_4'])
    
    

kidney_lake_bare = bare_surf.subdict['b7']
kidney_lake_snows = [snow_depth_surf12.subdict['2012_7'],
                    snow_depth_surf13.subdict['2013_7'],
                    snow_depth_surf15.subdict['2015_7'],
                    snow_depth_surf16.subdict['2016_7']]


#test = make_transect_obj(kidney_lake_bare, years, kidney_lake_snows, 270,70,750,200)
#test.PlotAll()

###

class Flux_point(object):
    '''A class for flux points where we determine the seasonal flux direction
    and flux amount.
    '''
    
    def __init__(self, x,y,name,windrange):
        
        self.results = dict()
        self.df = pd.DataFrame()
        self.x = x
        self.y = y
        self.name = name
        self.transect_keys = []
        self.windrange = np.arange(windrange[0],windrange[1],windrange[2])
        
        
        for w in self.windrange:
            self.transect_keys.append(self.name + '_' + str(w))
                    
        for k in zip(self.transect_keys, self.windrange):
            self.results[k[0]] = make_transect_obj(kidney_lake_bare,
                                years,
                                kidney_lake_snows, k[1],x,y,200) # 200 m long.
        
        idx = []
        tflux = []
        fluxerr = []
        fluxerrratio = []
        angle = []
    
        for k in self.results.items():
            idx.append(str(k[0]))
            angle.append(k[1].angle_deg)
            fluxerr.append(k[1].tabler_flux_err)
            tflux.append(k[1].tabler_flux)
            fluxerrratio.append(k[1].flux_err_ratio)
        
        self.df['idx']=idx
        self.df['wind direction'] = angle
        self.df['tabler flux error'] = fluxerr
        self.df['abs flux error'] = abs(self.df['tabler flux error'])
        self.df['tabler flux'] = tflux
        self.df['y'] = y
        self.df['err_ratio'] = fluxerrratio
        self.df['x']=x
        self.df.sort_values(['abs flux error'],inplace=True)
               
        tabler_match1 = self.df.iloc[0]
        tabler_match2 = self.df.iloc[1]
        tabler_match3 = self.df.iloc[2]
#        tabler_match1 = self.df.ix[np.argmin(abs(self.df['err_ratio']))]
        self.tabler_vector1 = tabler_match1['wind direction']
        self.tabler_vector2 = tabler_match2['wind direction']
        self.tabler_vector3 = tabler_match3['wind direction']

        
def plot_flux(points):
    '''
    Plot the profile lines on top of one snow depth map.
    Lines should fan out from each point at every angle specified.
    '''
    fig=plt.figure(figsize=(8,5))
    ax1=plt.subplot(1,1,1)
    im1=plt.imshow(kidney_lake_snows[2], cmap='viridis', vmin = 0, vmax = 2)
    ax1.set_ylabel('m',size = 7)
    ax1.set_xlabel('m',size = 7)
    plt.title('2015 Snow Depth Map [m]')                                            
    cbar_ax = fig.add_axes([0.90, 0.15, 0.05, 0.7])
    fig.colorbar(im1, cax=cbar_ax)

    for p in points:
        for k in p.results:
            ax1.plot([p.results[k].x1, p.results[k].x3],
                     [p.results[k].y1, p.results[k].y3], c='r',
                     alpha=0.5, linestyle='-', linewidth=1)
    #plt.savefig('/home/cparr/Snow_Patterns/figures/tabler_test.png',dpi=450)
        
def flux_map(points):
    '''
    Plot the profile lines on top of one snow depth map.
    Lines should fan out from each point at every angle specified.
    '''
    fig=plt.figure(figsize=(8,5))
    ax1=plt.subplot(1,1,1)
    im1=plt.imshow(kidney_lake_snows[2], cmap='viridis', vmin = 0, vmax = 2)
    ax1.set_ylabel('m',size = 7)
    ax1.set_xlabel('m',size = 7)
    plt.title('2015 Snow Depth Map [m]')                                            
    cbar_ax = fig.add_axes([0.90, 0.15, 0.05, 0.7])
    fig.colorbar(im1, cax=cbar_ax)

    for p in points:
        for k in p.results:
            if p.results[k].angle_deg == p.tabler_vector1 or \
                p.results[k].angle_deg == p.tabler_vector2 or \
                p.results[k].angle_deg == p.tabler_vector3:
                
                if p.results[k].angle_deg == p.tabler_vector1:
                    width=7.0
                    fc = 'r'
                elif p.results[k].angle_deg == p.tabler_vector2:
                    width=5.0
                    fc = 'y'
                else:
                    width = 3.0
                    fc='white'
                        
                ax1.arrow(
                        p.results[k].x1,            # x
                        p.results[k].y1,            # y
                        p.results[k].x3 - p.results[k].x1,            # dx
                        p.results[k].y3 - p.results[k].y1,            # dy
                        length_includes_head = True,
                        width = width,
                        fc = fc

                    )
#                ax1.annotate(str(int(p.results[k].mean_flux)),
#                             xy=(p.results[k].x1, p.results[k].y1),
#                                xycoords="data",
#                  va="center", ha="center",
#                  bbox=dict(boxstyle="round", fc="w"))
    #plt.savefig('/home/cparr/Snow_Patterns/figures/tabler_test.png',dpi=450)


lake_n = Flux_point(125,650,'lake_n',[180,370,15])
lake_w = Flux_point(80,750,'lake_w',[180,370,15])
lake_s = Flux_point(90,850,'lake_s',[180,370,15])
lake_n1 = Flux_point(135,670,'lake_n1',[180,370,15])
lake_w1 = Flux_point(90,770,'lake_w1',[180,370,15])
lake_s1 = Flux_point(100,870,'lake_s1',[180,370,15])

gully1_n = Flux_point(80,200,'gully1_n',[180,370,15])
gully1_w = Flux_point(80,250,'gully1_w',[180,370,15])
gully1_s = Flux_point(80,300,'gully1_s',[180,370,15])
gully1_n1 = Flux_point(95,200,'gully1_n1',[180,370,15])
gully1_w1 = Flux_point(95,250,'gully1_w1',[180,370,15])
gully1_s1 = Flux_point(110,300,'gully1_s1',[180,370,15])
gully1_n2 = Flux_point(110,200,'gully1_n2',[180,370,15])
gully1_w2 = Flux_point(110,250,'gully1_w2',[180,370,15])
gully1_s2 = Flux_point(110,300,'gully1_s2',[180,370,15])

gully2_n = Flux_point(310,50,'gully2_n',[180,370,15])
gully2_w = Flux_point(310,100,'gully2_w',[180,370,15])
gully2_s = Flux_point(310,150,'gully2_s',[180,370,15])
gully2_n1 = Flux_point(345,50,'gully2_n1',[180,370,15])
gully2_w1 = Flux_point(345,100,'gully2_w1',[180,370,15])
gully2_s1 = Flux_point(345,150,'gully2_s1',[180,370,15])

gully3_n = Flux_point(1100,250,'gully3_n',[180,370,15])
gully3_w = Flux_point(1100,300,'gully3_w',[180,370,15])
gully3_s = Flux_point(1050,350,'gully3_s',[180,370,15])
gully3_n1 = Flux_point(1100,280,'gully3_n1',[180,370,15])
gully3_w1 = Flux_point(1100,330,'gully3_w1',[180,370,15])
gully3_s1 = Flux_point(1050,380,'gully3_s1',[180,370,15])
gully3_n2 = Flux_point(1000,400,'gully3_n2',[180,370,15])
gully3_w2 = Flux_point(1000,415,'gully3_w2',[180,370,15])
gully3_s2 = Flux_point(1000,430,'gully3_s2',[180,370,15])
gully3_n3 = Flux_point(1075,500,'gully3_n3',[180,370,15])
gully3_w3 = Flux_point(1100,500,'gully3_w3',[180,370,15])
gully3_s3 = Flux_point(1125,500,'gully3_s3',[180,370,15])
gully3_n4 = Flux_point(1025,450,'gully3_n4',[180,370,15])
gully3_w4 = Flux_point(1050,450,'gully3_w4',[180,370,15])
gully3_s4 = Flux_point(1075,450,'gully3_s4',[180,370,15])

flux_pts = [lake_n,lake_w,lake_s,
            lake_n1,lake_w1,lake_s1,
          gully1_n,gully1_w,gully1_s,
          gully1_n1,gully1_w1,gully1_s1,
          gully1_n2,gully1_w2,gully1_s2,
          gully2_n,gully2_w,gully2_s,
          gully2_n1,gully2_w1,gully2_s1,
          gully3_n,gully3_w,gully3_s,
          gully3_n1,gully3_w1,gully3_s1,
          gully3_n2,gully3_w2,gully3_s2,
          gully3_n3,gully3_w3,gully3_s3,
          gully3_n4,gully3_w4,gully3_s4]

#plot_flux(flux_pts)
    
flux_map(flux_pts)

#plt.figure()
#sns.barplot(x=gully3_w4.df['wind direction'],y=gully3_w4.df['flux err. / tabler flux'])




####### everything below here is trying to follow the curve of the lake
#nd_limits = []
#for k in kidney_lake_results:
#    if kidney_lake_results[k].angle_deg == 290:
#        nd_limits.append(kidney_lake_results[k].no_data_limit)
#
##limit of lake
#
#curve_list = []
#ylocs = []
#xlocs = []
#
#i = 0
#
#for l in kidney_lake_bare:
#    
#    
#    nan_idx = np.argwhere(np.isnan(l[50:]))
#
#    if len(nan_idx) > 0:
#        c = min(nan_idx)
#        if c < 150:
#            ylocs.append(i)
#            curve_list.append(int(c))
#    i += 1
#
#for c in curve_list:
#    xlocs.append(c-50)
#    
#
#    
#plt.plot(curve_list)
#plt.plot(xlocs)
#plt.plot(ylocs)
## test using lake buffer as start points
#
#kidney_lake_results = dict()
#df = pd.DataFrame()
#
#def test_wind_dirs(dir1,dir2,step):
#
#    profile_keys = []
#    wind_dirs = []
#    
#    for n in np.arange(dir1,dir2,step):
#        for i in range(1,12):
#            profile_keys.append('Lake_'+str(n)+'_'+str(i))
#            wind_dirs.append(n)
# 
#    start_xs = [n for n in xlocs]*len(profile_keys)
#    start_ys = [n for n in np.arange(min(ylocs),max(ylocs),1)]*len(profile_keys)
#
#
#    
#    for k in zip(profile_keys, start_xs, start_ys, wind_dirs):
#        kidney_lake_results[k[0]] = make_transect_obj(kidney_lake_bare,
#                            years,
#                            kidney_lake_snows,
#                            k[3],k[1],k[2],200,50,106)
#        
#    idx = []
#    tflux = []
#    fluxerr = []
#    angle = []
#    ystart = []
#    
#    for k in kidney_lake_results.items():
#        idx.append(str(k[0]))
#        angle.append(k[1].angle_deg)
#        fluxerr.append(k[1].tabler_flux_err)
#        tflux.append(k[1].tabler_flux)
#        ystart.append(k[1].y1)
#        
#    df['idx']=idx
#    df['wind direction'] = angle
#    df['tabler flux error'] = fluxerr
#    df['tabler flux'] = tflux
#    df['y1'] = ystart
#        
#test_wind_dirs(260,305,5)
#
#plot_all_one_map()
#wt0 = Transect(watertrack_lake_bare,years,watertrack_lake_snows, 300,200,300,250)
#wt0.TablerProfile(23,48, wt0.snowdict['2012']['winter surface profile'], 8)
#big_lake0 = Transect(big_lake_bare,years,big_lake_snows, 300,40,340,120)
#big_lake0.TablerProfile(30,70, big_lake0.snowdict['2012']['winter surface profile'], 8)

#==============================================================================
#  
#     def savitzky_golay(self, y, window_size, order, deriv=0, rate=1):
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
# 
#     def smooth_surfaces(self):
#         ''' Use Savitzky-Golay to generate slope and curvature'''
#         self.smooth_bare_earth = self.savitzky_golay(self.bare_earth, 9, 1)
#         self.slope = self.savitzky_golay(self.smooth_bare_earth, 9, 1, 1)
#         self.curvature = self.savitzky_golay(self.slope, 9, 1, 1)
#     
#     def thresh_data(self, arr, sig, thresh):
#         ''' Use Gaussian Filter to create boolean array where True values are
#         consecutively greater than some threshold. This is useful where there
#         are many drifts arranged in series, e.g. watertracks.
#         '''    
#         sigma = sig
#         threshold = thresh
#         self.above_threshold = gaussian_filter(arr, sigma=sigma) > threshold
# 
#     def contiguous_regions(self, condition):
#         """Finds contiguous True regions of the boolean array "condition". Returns
#         a 2D array where the first column is the start index of the region and the
#         second column is the end index."""
#     
#         # Find the indicies of changes in "condition"
#         d = np.diff(condition)
#         idx, = d.nonzero() 
#     
#         # We need to start things after the change in "condition". Therefore, 
#         # we'll shift the index by 1 to the right.
#         idx += 1
#     
#         if condition[0]:
#             # If the start of condition is True prepend a 0
#             idx = np.r_[0, idx]
#     
#         if condition[-1]:
#             # If the end of condition is True, append the length of the array
#             idx = np.r_[idx, condition.size] # Edit
#     
#         # Reshape the result into two columns
#         idx.shape = (-1,2)
#         self.drift_index = idx
#     
#     def drift_brackets(self):
#         ''' Define the start and end of a drift based on some kind of 
#         topographic indicator. These might need to vary depending on the type
#         of terrain'''
#         self.min_curve_start = np.nanargmin(self.curvature)
#         self.max_curve_end = np.nanargmax(self.curvature)
#         
#     def measure_flux(self, start, end):
#         ''' Use trapezoidal rule integration to calculate flux over the drift
#         fetch. Values will be m^3 per lineal m, so essentialy we have m^2 which
#         makes sense because we are getting the area under the snow depth curve.
#         We divide the initial result by 2 because our data cells are 2m x 2m.
#         Parameters are start and end of drift zone for integration.
#         '''
#         self.flux = integrate.simps(self.snow_depth, range(start, end)) / 2.0
#         #self.flux = np.trapz(self.snow_depth[start:end]) / 2.0
#         self.drift_length = end - start
#         self.avg_slope_under_drift = np.nanmean(self.slope[start:end])
# 
# 
# ############
# def make_transect(bsubsurf, ssubsurf, tline):
#     
#     tname = Transect(bare_surf.subdict[bsubsurf][tline], 
#                      snow_depth_surf.subdict[ssubsurf][tline])
#     tname.thresh_data(tname.snow_depth,3,np.nanmean(tname.snow_depth))
#     tname.smooth_surfaces()
#     tname.contiguous_regions(tname.above_threshold == True)
#     
#     return tname
#     
#     # uncomment below if entire transect is a drift
#     #t1.drift_brackets()
#     #t1.measure_flux(t1.min_curve_start,t1.max_curve_end)
#     
# def make_subdrifts(tname):
# 
#     dct = dict()
#     # I expect to see meaningless 'mean of empty slice
#     # RuntimeWarnings in this block
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore", category=RuntimeWarning)        
#         for d in tname.drift_index:
#             dr_id = str(d)[1:-1]
#             dct[dr_id] = Subdrift(tname, d)
#             dct[dr_id].smooth_surfaces()
#             dct[dr_id].measure_flux(dct[dr_id].start,
#                       dct[dr_id].end)
#     return dct
# 
# # this is cool!
# # big flux...low flux...big flux..low flux..then decreasing trend:
# def scatter_flux(drift_dct):
#     
#     fluxes = []
#     starts = []
#     dlengths = []
#     for k,v in drift_dct.items():
#         fluxes.append(v.flux)
#         starts.append(v.start)
#         dlengths.append(v.drift_length)
#     #plt.figure()
#     plt.scatter(starts,fluxes,s=dlengths)
#     plt.ylabel('Drift Flux')
#     plt.xlabel('drift start [m]')
#     return fluxes
# 
# def plot_overview(im, dct, tline):
#     
#     dims = np.isnan(im[tline])
#     x = np.diff(np.where(dims))
#     cutoff = np.where(x>1)
#     include_start = cutoff[1][0]
#     include_stop = x.max()
#     
#     plt.figure()
#     plt.imshow(im[:,include_start:include_stop])
#     
#     for k,v in dct.items():
#         x1 = v.start / im[:,include_start:include_stop].shape[1]
#         x2 = v.end / im[:,include_start:include_stop].shape[1]
#         plt.axhline(y=tline, xmin = x1, xmax = x2, c = 'r', alpha=0.5)
# 
# ##############################################################################
# tlines = np.arange(220,455,5)
# tnames = []
# dnames=[]
# for y in tlines:
#     t = 't5_' + str(y)
#     d = 't5_' + str(y) +'_sub'
#     tnames.append(t)
#     dnames.append(d)
#     del y, d, t
# 
# big_t_dct = dict()
# for f in zip(tnames ,tlines):
#     big_t_dct[f[0]] = make_transect('b5', 's5', f[1])
# 
# big_s_dct = dict()
# for f in zip(dnames ,tnames):
#     big_s_dct[f[0]] = make_subdrifts(big_t_dct[f[1]])
# 
# all_flux = []
# flux_sums = []
# for k in big_s_dct.keys():
#     f = scatter_flux(big_s_dct[k])
#     all_flux.append(f)
# 
# for f in all_flux:
#     flux_sums.append(sum(f) / 2)
# 
# plt.scatter(tlines, flux_sums)
# 
# 
# for k in zip(big_s_dct.keys(), tlines):
#     plot_overview(snow_depth_surf.subdict['s5'], big_s_dct[k[0]], k[1])
# 
# ################################################################