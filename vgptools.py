# -*- coding: utf-8 -*-
"""
Created on Mon Feb 03 10:49:00 2025

@author: paselkin
"""
## Packages
import pmagpy
import pmagpy.pmagplotlib as pmagplotlib
import pmagpy.ipmag as ipmag
import pmagpy.pmag as pmag
import pmagpy.contribution_builder as cb
from pmagpy import convert_2_magic as convert
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import glob as glob
import shutil
import math
from pathlib import Path
import time
from tqdm import tqdm
import copy

## Functions

def strike_to_dip_direction(df, column):
    """
    Converts a column of strike azimuth data to dip direction. 
    Suitable for MagIC dip_direction. 
    Uses right-hand rule convention (strike + 90 = dip direction).
    
    Parameters
    ----------
    df: An input dataframe
    column: The name of a column in that dataframe
    
    Returns
    -------
    result: A Pandas series with the same number of rows as df 
        containing dip direction (in degrees, 0-360) in each row. 
    """
    result=(df[column]+90.)%360.
    return(result)

def result_quality_criterion(df, column):
    """
    Checks a column of the input data for values. 
    Rows where no value exists are labeled 'b' (bad);
    rows where a value exists are labeled 'g' (good).
    Suitable for MagIC result_quality.
    
    Parameters
    ----------
    df: An input dataframe
    column: The name of a column in that dataframe
    
    Returns
    -------
    result: A Pandas series with the same number of rows as df 
        containing 'g' or 'b' in each row. 
    """
    result=df.apply(lambda x: 'b' if np.isnan(x[column]) else 'g',axis=1)
    return(result)

def tilt_correct(input_df,digits=1):
    """
    Tilt-correct a samples or sites dataframe. Removes 100% of tilt by
    rotating layers about strike axis (pmagpy.dotilt).
    
    Parameters
    ----------
    input_df: An input dataframe with columns that follow the MagIC 
        Data Model. Must contain dir_dec, dir_inc, bed_dip, and 
        bed_dip_direction.
    digits (optional): Number of digits after decimal to round results.
        Default: 1
    
    Returns
    -------
    result_df: A dataframe with dir_dec and dir_inc corrected for tilt.
        dir_tilt_correction set to 1.
    """
    tc_dec=np.zeros(len(input_df))
    tc_inc=np.zeros(len(input_df))
    for i in range(len(input_df)):
        tc_dec[i],tc_inc[i] = pmagpy.pmag.dotilt(input_df['dir_dec'][i],input_df['dir_inc'][i],
                                                 input_df['bed_dip_direction'][i],input_df['bed_dip'][i])

    result_df = copy.copy(input_df)
    result_df['dir_dec']=tc_dec
    result_df['dir_dec']=result_df['dir_dec'].round(digits)
    result_df['dir_inc']=tc_inc
    result_df['dir_inc']=result_df['dir_inc'].round(digits)
    result_df['dir_tilt_correction']=1
    return(result_df)

def calculate_vgps(input_df,digits=1):
    """
    Calculate virtual geomagnetic poles for a dataframe with MagIC
    format directional data (pmagpy.dia_vgp).
    
    Parameters
    ----------
    input_df: An input dataframe with columns that follow the MagIC 
        Data Model. Must contain dir_dec, dir_inc, dir_alpha95, lat, 
        and lon.
    digits (optional): Number of digits after decimal to round results.
        Default: 1
    
    Returns
    -------
    result_df: The input dataframe with vgp_lat, vgp_lon, vgp_dp, and vgp_dm
    added as columns.
    """
    # Outputs dataframe with columns: vgp_lat, vgp_lon, vgp_dp, vgp_dm
    data=input_df[['dir_dec','dir_inc','dir_alpha95','lat','lon']].to_numpy()
    vgps=np.array(pmag.dia_vgp(data)).round(digits) 
    vgps=vgps.transpose()
    vgps_df=pd.DataFrame(vgps,columns=['vgp_lon','vgp_lat','vgp_dp','vgp_dm'],index=input_df.index)
    result_df=pd.concat([input_df,vgps_df],axis=1)
    return(result_df)

def flipdirs(dec,inc,princ_dec,princ_inc):
    """
    Helper function to flip hemisphere of dec,inc pairs depending on their angle from principal direction
    """
    n = len(dec)
    #angle = np.zeros(n) # to store the angle between the direction and the principal direction
    flipped_dec = np.zeros(n)
    flipped_inc = np.zeros(n)
    for i in range(n):
        angle = pmag.angle([dec[i],inc[i]],[princ_dec,princ_inc]) 

        if angle > 90:
            flipped_dec[i] = (dec[i] + 180.) % 360.
            flipped_inc[i] = - inc[i]
        else:
            flipped_dec[i] = dec[i]
            flipped_inc[i] = inc[i]
    return flipped_dec, flipped_inc
    
def to_di_block(df):
    """
    Helper function to transform dataframe to di_block (list of dec, inc, magnitude)
    """
    df['magnitude']=1.0 # Add this in because Pmagpy assumes that vectors in di_blocks have a magnitude
    di_block = df[['dir_dec','dir_inc','magnitude']].apply(lambda x: x.to_list(),axis=1).to_list()
    return(di_block)


def group_average(dir_df, nb=500, column_map=None, flip=True, return_distribution=False):
    """
    Performs a bootstrap for direction DataFrame (group of sites, samples, etc.) with parametric bootstrap,
    providing bootstrap kappa parameter

    Parameters
    ----------
    dir_df : Pandas DataFrame with columns:
        dir_dec : mean declination
        dir_inc : mean inclination
        dir_n : number of data points in mean
        dir_k : Fisher k statistic for mean
    nb : number of bootstraps, default is 500; if np = 0, do not bootstrap (instead, take Fisher average of input data)
    flip : whether to flip polarity of input directions
        False : Do not flip (assumes all magnetization vectors are normal)
        True : Flip so all directions are aligned with principal component - Default
    return_distribution : return DataFrame of dec, inc, kappa values
            (default: False)

    Returns
    -------
    if return_distribution is False:
        boot_results: Pandas DataFrame with columns:
            dir_dec : bootstrapped median declination value
            dir_inc : bootstrapped median inclination value
            dir_k : bootstrapped median kappa value
            dir_n : number of rows in original dataframe
            dir_alpha95: A95 associated with median kappa
    if return_distribution is True:
        boot_results: Pandas DataFrame with all bootstrapped values of columns above
    
  
    """
    dir_df=dir_df.dropna()
    N = dir_df.shape[0] # Note: N counts only non-NaN elements
    if N>1:
        if column_map is not None:
            dir_df=dir_df.rename(columns=column_map)
        if flip is True:
            dir_array=dir_df[['dir_dec','dir_inc']].to_numpy()
            pc=pmag.doprinc(dir_array)
            flipped_dir_dec, flipped_dir_inc=flipdirs(dir_array[:,0],dir_array[:,1],pc['dec'],
                                                      pc['inc'])
            dir_df['dir_dec']=flipped_dir_dec
            dir_df['dir_inc']=flipped_dir_inc
        all_boot_results = pd.DataFrame(np.zeros((nb,5)),columns=
                                        ['dir_dec','dir_inc','dir_n','dir_k','dir_alpha95'])
        if (nb>0):
            for k in tqdm(range(nb)):
                boot_di=dir_df.apply(
                    lambda x:np.array(list(ipmag.fishrot(k=x['dir_k'],
                                                         n=int(x['dir_n']),
                                                         dec=x['dir_dec'],
                                                         inc=x['dir_inc'],
                                                         di_block=False))
                                     ).T,axis=1).explode().apply(
                    lambda x: pd.Series(
                        {'dir_dec':x[0],'dir_inc':x[1]}
                    )
                )
                all_boot_results.iloc[k,:] = pd.Series(
                    pmag.dir_df_fisher_mean(boot_di)).drop(index=['csd','r']
                                                          ).rename(
                    index={'alpha95':'dir_alpha95',
                           'dec':'dir_dec',
                           'inc':'dir_inc',
                           'k':'dir_k',
                           'n':'dir_n'}
                )

            b = 20.**(1./(N -1.))-1.
            if return_distribution:
                results = all_boot_results
                results['a']=1-b*(N-1.)/((N*(results['dir_k']-1.))-1.)
                results['dir_n']=N
                results['dir_alpha95']=np.degrees(np.arccos(results['a']))
                results['result_type']='a'
            else:
                fisher_avg=pd.DataFrame([pmag.dir_df_fisher_mean(all_boot_results)]).rename(columns={'alpha95':'dir_alpha95',
                                          'dec':'dir_dec',
                                          'inc':'dir_inc',
                                          'k':'dir_k',
                                          'n':'dir_n'})
                fisher_avg['dir_k'] = np.median(all_boot_results['dir_k'].to_numpy())
                fisher_avg['dir_n'] = N
                a=1-b*(N-1.)/((N*(fisher_avg['dir_k']-1.))-1.)
                fisher_avg['dir_alpha95'] = np.degrees(np.arccos(a))
                results=fisher_avg[['dir_dec','dir_inc','dir_k','dir_n','dir_alpha95']]
                results['result_type']='a'
        else:
            fisher_avg=pd.DataFrame([pmag.dir_df_fisher_mean(dir_df)]).rename(columns={'alpha95':'dir_alpha95',
                                      'dec':'dir_dec',
                                      'inc':'dir_inc',
                                      'k':'dir_k',
                                      'n':'dir_n'})
            results=fisher_avg[['dir_dec','dir_inc','dir_k','dir_n','dir_alpha95']]
            results['result_type']='a'
    else:
        results=dir_df[['dir_dec','dir_inc','dir_k','dir_alpha95']]
        results['dir_n']=1
        results['result_type']='i'
    if column_map is not None:
        column_map_r=dict(zip(column_map.values(),column_map.keys()))
        results=results.rename(columns=column_map_r)
    return(results)


def roll_up_sites_locations(sites_df, location_parameter='location', 
                            location_bbox=True, age=False, nb=500, flip=True,
                            digits=1):
    """
    'Rolls up' a dataframe of sites into a locations dataframe that contains average 
    directional and vgp information for each location. 

    Parameters
    ----------
    sites_df : MagIC format sites DataFrame
    location_parameter: Name of column with location information - Default: 'location'
    location_bbox: If True (default), uses bounding box defined by max and min
        of lat and lon data. If False, uses arithmetic average lat and lon (because
        presumably the sites are all the same lat and lon...). If False and the input 
        dataframe has no coordinate info, leaves the coordinates of the locations blank.
    age: If False (default), does not define age, otherwise uses range of age values.
    nb : number of bootstraps, default is 500; if np = 0, do not bootstrap 
        (instead, take Fisher average of input data)
    flip : whether to flip polarity of input directions
        False : Do not flip (assumes all magnetization vectors are normal)
        True : Flip so all directions are aligned with principal component - Default
    digits: Number of digits for rounding (default 1)
    

    Returns
    -------
    locations_df: Pandas DataFrame with locations data
    """
    sample_map={'dir_n_samples':'dir_n'} 
    locations_df=pd.DataFrame([])
    sites_group=sites_df.groupby(location_parameter)
    locations_df['location']=sites_group.groups.keys()
    # What are all the sites in this location?
    locations_df['sites']=sites_group['site'].unique().apply(':'.join).reset_index(drop=True) 
    if 'lithologies' in sites_group.obj.columns:
        # What are all the lithologies in this location?
        locations_df['lithologies']=sites_group['lithologies'].unique().apply(':'.join).reset_index(drop=True) 
    if location_bbox:
        # The following 4 lines find the bounding box of coordinates for each grouping of sites. 
        locations_df['lat_s']=sites_group['lat'].min().reset_index(drop=True) 
        locations_df['lat_n']=sites_group['lat'].max().reset_index(drop=True)
        locations_df['lon_e']=sites_group['lon'].max().reset_index(drop=True)
        locations_df['lon_w']=sites_group['lon'].min().reset_index(drop=True)
    elif 'lon' in sites_group.obj.columns:
        locations_df['lat']=sites_group['lat'].mean().reset_index(drop=True)
        locations_df['lon']=sites_group['lon'].mean().reset_index(drop=True)
    if age:
        locations_df1['age_low']=sites_group['age'].min().reset_index(drop=True)
        locations_df1['age_high']=sites_group['age'].max().reset_index(drop=True)
    # The following does a parametric bootstrap for each location to find the average kappa and a95 values
    locations_df=pd.concat([locations_df,sites_group.apply(
        lambda x: group_average(x,nb=nb,flip=flip,column_map=sample_map)
        ).reset_index()],axis=1)
    # Need to rename columns: now our "n" is numbers of sites, since locations_df is the locations dataframe (a site-level average).
    locations_df=locations_df.rename({'dir_n_samples':'dir_n_sites'})
    # Remove duplicate column names
    locations_df = locations_df.loc[:,~locations_df.columns.duplicated()].copy()
    # Remove "level 0" column
    if 'level_1' in locations_df.columns:
        locations_df.drop('level_1',axis=1,inplace=True)
    locations_df=locations_df.round(digits)
    return(locations_df)