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
    
def dir_df_boot_ci(dir_df, nb=500, column_map=None, return_distribution=False):
    """
    Performs a bootstrap for direction DataFrame with parametric bootstrap,
    providing bootstrap kappa parameter

    Parameters
    ----------
    dir_df : Pandas DataFrame with columns:
        dir_dec : mean declination
        dir_inc : mean inclination
        dir_n : number of data points in mean
        dir_k : Fisher k statistic for mean
    nb : number of bootstraps, default is 5000
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
            dir_alpha95: A95 associated with medioan kappa
    if return_distribution is True:
        boot_results: Pandas DataFrame with all bootstrapped values of columns above
    
  
    """
    dir_df=dir_df.dropna()
    N = dir_df.shape[0] # Note: N counts only non-NaN elements
    if N>1:
        if column_map is not None:
            dir_df=dir_df.rename(columns=column_map)
        #dir_df,=
        all_boot_results = pd.DataFrame(np.zeros((nb,5)),columns=['dir_dec','dir_inc','dir_n','dir_k','dir_alpha95'])
        t0=time.time()
        for k in tqdm(range(nb)):
            boot_di=dir_df.apply(lambda x: np.array(list(ipmag.fishrot(k=x['dir_k'],n=int(x['dir_n']),dec=x['dir_dec'],inc=x['dir_inc'],di_block=False))).T,axis=1).explode().apply(lambda x: pd.Series({'dir_dec':x[0],'dir_inc':x[1]}))
            all_boot_results.iloc[k,:] = pd.Series(pmag.dir_df_fisher_mean(boot_di)).drop(index=['csd','r']).rename(index={'alpha95':'dir_alpha95',
                                      'dec':'dir_dec',
                                      'inc':'dir_inc',
                                      'k':'dir_k',
                                      'n':'dir_n'})
        t1=time.time()
        t_total=t1-t0

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
            fisher_avg['time']=t_total
            results=fisher_avg[['dir_dec','dir_inc','dir_k','dir_n','dir_alpha95','time']]
            results['result_type']='a'
    else:
        results=dir_df[['dir_dec','dir_inc','dir_k','dir_alpha95']]
        results['dir_n']=1
        results['time']=0
        results['result_type']='i'
    if column_map is not None:
        column_map_r=dict(zip(column_map.values(),column_map.keys()))
        results=results.rename(columns=column_map_r)
    return(results)

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
            flipped_dir_dec, flipped_dir_inc=flipdirs(dir_array[:,0],dir_array[:,1],pc[0],pc[1])
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

def add_vgps(sites_df):
    # Outputs dataframe with columns: vgp_lat, vgp_lon, vgp_dp, vgp_dm
    data=sites_df[['dir_dec','dir_inc','dir_alpha95','lat','lon']].to_numpy()
    vgps=np.array(pmag.dia_vgp(data)) 
    vgps=vgps.transpose()
    return(pd.DataFrame(vgps,columns=['vgp_lon','vgp_lat','vgp_dp','vgp_dm'],index=sites_df.index))

def to_di_block(df):
    df['magnitude']=1.0 # Add this in because Pmagpy assumes that vectors in di_blocks have a magnitude
    di_block = df[['dir_dec','dir_inc','magnitude']].apply(lambda x: x.to_list(),axis=1).to_list()
    return(di_block)

def dataframe_flip(dir_df, how='norm_nh'):
    """
    Flips directions in dataframe given assumptions listed.
    
    Parameters
    ----------
    dir_df : Pandas DataFrame with columns:
        dir_dec : mean declination
        dir_inc : mean inclination
        dir_n : number of data points in mean
        dir_k : Fisher k statistic for mean
    how : assumptions to use when flipping polarity of input directions
        'norm_nh' : flip so all directions are aligned with normal assuming northern hemisphere (N/down)
        'norm_sh' : flip so all directions are aligned with normal assuming southern hemisphere (S/down)
        'rev_nh' : flip so all directions are aligned with reversed assuming northern hemisphere (S/up)
        'rev_sh' : flip so all directions are aligned with reversed assuming southern hemisphere (N/up)
        'trim_norm_nh', 'trim_norm_sh', etc. : exclude directions that are >90 degrees normal/reversed in specified hemisphere
    
    Returns
    -------
    boot_results: Pandas DataFrame with columns:
            dir_dec : bootstrapped median declination value
            dir_inc : bootstrapped median inclination value
            dir_k : bootstrapped median kappa value
            dir_n : number of rows in original dataframe
            dir_alpha95: A95 associated with median kappa
    """
    # NOTE: THIS IS A PLACEHOLDER FOR RACHEL'S FUNCTION
    if how!='norm_nh':
        print ("Unimplemented Command Error.")
    return(dir_df)