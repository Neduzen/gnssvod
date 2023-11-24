"""
preprocess reads files and returns analysis-ready DataSet

pair_obs merges and pairs observations from sites according to specified pairing rules over the desired time intervals
"""
# ===========================================================
# ========================= imports =========================
import os
import time
import glob
import datetime
import numpy as np
import pandas as pd
import xarray as xr
import multiprocessing
from gnssvod.io.readFile import read_obsFile
from gnssvod.funcs.checkif import (isfloat, isint, isexist)
from gnssvod.funcs.date import doy2date
from gnssvod.position.interpolation import sp3_interp_fast
from gnssvod.position.position import gnssDataframe
from gnssvod.funcs.constants import _system_name
import pdb
# ===========================================================

#-------------------------------------------------------------------------
#----------------- FILE SELECTION AND BATCH PROCESSING -------------------
#-------------------------------------------------------------------------
def preprocess(filepattern,orbit=True,interval=None,keepvars=None,nworkers=None,outputdir=None):
    """
    Returns lists of Observation objects containing GNSS observations read from RINEX observation files
    
    Parameters
    ----------
    filepattern: dictionary 
        Dictionary of station names and UNIX-style patterns to match RINEX 
        observation files. For example filepattern={'station1':'/path/to/files/of/station1/*O'}
    
    orbit: bool (optional) 
        if orbit=True, will download orbit solutions and calculate Azimuth and Elevation parameters
        if orbit=False, will not calculate additional gnss parameters
        
    interval: string or None (optional)
        if interval = None, the observations will be returned at the same rate as they were saved
        if interval = pandas Timedelta or str, this will be used to resample (average) the obervations (e.g. interval="15S")
    
    keepvars: list of strings or None (optional)
        Defines what columns are kept after processing. This can help reduce the size of the saved data.
        For example keepvars = ['S1','S2','Azimuth','Elevation']
        If None, no columns are removed
        
    nworkers: int or None (optional)
        if int, will open a multiprocessing pool with nworkers to read the obs files in parallel
        
    outputdir: dictionary (optional)
        A dictionary of station names and folders indicating where to save the preprocessed data
        For example outputdir={'station1':'/path/where/to/save/preprocessed/data'}
        Dictionary keys must be the same as in the filepattern argument
        Data will be saved as a netcdf file, recycling the original file name
        If this argument is None, data will be returned in a list but not saved
        
    Returns
    -------
    Dictionary of station names associated with a list of xarray Datasets containing the data from each file
    For example output={'station1':[gnssvod.io.io.Observation,gnssvod.io.io.Observation,...]}
    
    """
    # grab all files matching the patterns
    filelist = get_filelist(filepattern)
    
    out = dict()
    for item in filelist.items():
        station_name = item[0]
        filelist = item[1]
        
        # open a pool to read and resample files in parallel if requested
        if nworkers:
            print(f"Starting parallel processing Pool with {nworkers} workers")
            pool = multiprocessing.Pool(nworkers)
            result = pool.map(read_obsFile,filelist)
            if interval:
                result = pool.starmap(resample_obs,[(x,interval) for x in result])
            pool.close()
        # else process files iteratively
        else:
            result = [read_obsFile(x) for x in filelist]
            if interval:
                result = [resample_obs(x,interval) for x in result]
            
        # for each pandas Observation dataframe iteratively
        for i,x in enumerate(result):
            if orbit:
                print(f"Calculating Azimuth and Elevation ({i+1} out of {len(result)})")
                # note: orbit cannot be parallelized easily because it 
                # downloads and unzips third-party files in the current directory
                if i==0:
                    # on the first iteration, the orbit data is returned as well
                    x, orbit_data = add_azi_ele(x)
                else:
                    # on following iterations the orbit data is tentatively recycled to reduce computational time
                    x, orbit_data = add_azi_ele(x, orbit_data)
            # make sure we drop any duplicates
            x.observation=x.observation[~x.observation.index.duplicated(keep='first')]
            # only keep required vars and drop potential empty rows
            if keepvars is not None:
                x.observation = x.observation[keepvars].dropna(how='all')
                x.observation_types = keepvars
            # store result
            result[i]=x
        out[station_name]=result
    
    # output the files
    if outputdir:
        for item in out.items():
            # recover list of observation objects and output directory
            station_name = item[0]
            list_of_obs = item[1]
            ioutputdir = outputdir[station_name]
            # check that the output directory exists for that station
            if not os.path.exists(ioutputdir):
                os.makedirs(ioutputdir)
            for obs in list_of_obs:
                ds = obs.observation.to_xarray()
                ds.attrs['filename'] = obs.filename
                ds.attrs['observation_types'] = obs.observation_types
                ds.attrs['epoch'] = obs.epoch.isoformat()
                ds.attrs['approx_postion'] = obs.approx_position
                ds_name = os.path.splitext(os.path.basename(ds.attrs['filename']))[0]+'.nc'
                ds.to_netcdf(os.path.join(ioutputdir,ds_name))
            print(f"Saved {len(list_of_obs)} processed files in {ioutputdir}")
                
    return out

def resample_obs(obs,interval):
    obs.observation = obs.observation.groupby([pd.Grouper(freq=interval, level='Epoch'),pd.Grouper(level='SV')]).mean()
    obs.observation['epoch'] = obs.observation.index.get_level_values('Epoch')
    obs.observation['SYSTEM'] = _system_name(obs.observation.index.get_level_values("SV"))
    obs.interval = pd.Timedelta(interval).seconds
    return obs

def add_azi_ele(obs, orbit_data=None):
    if orbit_data is None:
        do = True
    elif (orbit_data.my_epoch==obs.epoch) and (orbit_data.my_interval==obs.interval):
        # if the orbit for the day corresponding to the epoch and interval is the same as the one that was passed, just reuse it. This drastically reduces the number of times orbit files have to be read.
        do = False
    else:
        do = True
    
    if do:
        # read (=usually download) orbit data
        orbit = sp3_interp_fast(obs.epoch, interval=obs.interval)
        # prepare an orbit object as well
        orbit_data = orbit
        orbit_data.my_epoch = obs.epoch
        orbit_data.my_interval = obs.interval
    else:
        orbit = orbit_data
    
    # calculate the gnss parameters (including azimuth and elevation)
    gnssdf = gnssDataframe(obs,orbit)  
    # add the gnss parameters to the observation dataframe
    obs.observation = obs.observation.join(gnssdf[['Azimuth','Elevation']])
    return obs, orbit_data

def get_filelist(filepatterns):
    if not isinstance(filepatterns,dict):
        raise Exception(f"Expected the input of get_filelist to be a dictionary, got a {type(filepatterns)} instead")
    filelists = dict()
    for item in filepatterns.items():
        station_name = item[0]
        search_pattern = item[1]
        flist = glob.glob(search_pattern)
        if len(flist)==0:
            raise Warning(f"Could not find any files matching the pattern {search_pattern}")
        filelists[station_name] = flist
    return filelists


#--------------------------------------------------------------------------
#----------------- PAIRING OBSERVATION FILES FROM SITES -------------------
#-------------------------------------------------------------------------- 

def pair_obs(filepattern,pairings,timeintervals,keepvars=None,outputdir=None):
    """
    Merges and pairs observations from sites according to specified pairing rules over the desired time intervals
    
    Parameters
    ----------
    filepattern: dictionary 
        Dictionary of station names and UNIX-style patterns to find the preprocessed NetCDF files 
        observation files. For example filepattern={'station1':'/path/to/files/of/station1/*.nc',
                                                    'station2':'/path/to/files/of/station2/*.nc'}
    
    pairings: dictionary 
        Dictionary of case names associated to a tuple of station names indicating which stations to pair, 
        with the reference station given first.
        For example pairings={'case1':('station1','station2')} will take 'station1' as the reference station.
        If data is to be saved, the case name will be taken as filename.
        
    timeintervals: pandas fixed frequency IntervalIndex
        The time interval(s) over which to pair data
        For example timeperiod=pd.interval_range(start=pd.Timestamp('1/1/2018'), periods=8, freq='D') will pair 
        data for each of the 8 days in timeperiod and return one DataSet for each day.
        
    keepvars: list of strings or None (optional)
        Defines what columns are kept after pairing is made. This helps reduce the size of the saved paired data.
        For example keepvars = ['S1_ref','S1_grn','S2_ref','S2_grn','Azimuth_grn','Elevation_grn']
        If None, no columns are removed
        
    outputdir: dictionary (optional)
        A dictionary of station names and folders indicating where to save the preprocessed data
        For example outputdir={'case1':'/path/where/to/save/paired/data'}
        Data will be saved as a netcdf file, the dictionary has to be consistent with the 'pairings' argument
        If this argument is None, data will not be saved
        
    Returns
    -------
    Dictionary of case names associated with a list of xarray Dataset(s) containing the paired
    data for each time interval contained in the 'timeperiod' argument.
    
    """
    out=dict()
    for item in pairings.items():
        case_name = item[0]
        print(f'Processing {case_name}')
        print(f'Listing the files matching with the interval')
        ref_name = item[1][0]
        grn_name = item[1][1]
        overall_interval = pd.Interval(left=timeintervals.min().left,right=timeintervals.max().right)
        # get all files
        ref_files = get_filelist({ref_name:filepattern[ref_name]})
        grn_files = get_filelist({grn_name:filepattern[grn_name]})
        # get Epochs from all files
        ref_epochs = [xr.open_mfdataset(x).Epoch for x in ref_files[ref_name]]
        grn_epochs = [xr.open_mfdataset(x).Epoch for x in ref_files[ref_name]]
        # check which files have data that overlaps with the desired time intervals
        ref_isin = [overall_interval.overlaps(pd.Interval(left=pd.Timestamp(x.values.min()),
                                                          right=pd.Timestamp(x.values.max()))) for x in ref_epochs]
        grn_isin = [overall_interval.overlaps(pd.Interval(left=pd.Timestamp(x.values.min()),
                                                          right=pd.Timestamp(x.values.max()))) for x in grn_epochs]
        print(f'Found {sum(ref_isin)} files for {ref_name} and {sum(grn_isin)} for {grn_name}')
        print(f'Reading')
        # open those files and convert them to pandas dataframes
        ref_data = [xr.open_mfdataset(x).to_dataframe() for x in np.array(ref_files[ref_name])[ref_isin]]
        grn_data = [xr.open_mfdataset(x).to_dataframe() for x in np.array(grn_files[grn_name])[grn_isin]]
        # concatenate, drop duplicates and sort the dataframe
        ref_data = pd.concat(ref_data)
        ref_data = ref_data[~ref_data.index.duplicated()].sort_index(level=['Epoch','SV'])
        grn_data = pd.concat(grn_data)
        grn_data = grn_data[~grn_data.index.duplicated()].sort_index(level=['Epoch','SV'])
        # inner join the two stations
        print(f'Pairing')
        iout = ref_data.join(grn_data,how='inner',lsuffix='_ref',rsuffix='_grn')
        # only keep required vars and drop potential empty rows
        if keepvars is not None:
            iout = iout[keepvars].dropna(how='all')
        # split the dataframe into multiple dataframes according to timeintervals
        out[case_name] = [x for x in iout.groupby(pd.cut(iout.index.get_level_values('Epoch').tolist(), timeintervals))]
        
    # output the files
    if outputdir:
        for item in out.items():
            # recover list of dataframes and output directory
            case_name = item[0]
            list_of_dfs = item[1]
            ioutputdir = outputdir[case_name]
            # check that the output directory exists for that station
            if not os.path.exists(ioutputdir):
                os.makedirs(ioutputdir)
            print(f'Saving files for {case_name} in {ioutputdir}')
            for df in list_of_dfs:
                # make timestamp for filename in format yyyymmddhhmmss_yyyymmddhhmmss
                ts = f"{df[0].left.strftime('%Y%m%d%H%M%S')}_{df[0].right.strftime('%Y%m%d%H%M%S')}"
                filename = f"{case_name}_{ts}.nc"
                # convert dataframe to xarray for saving to netcdf
                ds = df[1].to_xarray()
                ds.to_netcdf(os.path.join(ioutputdir,filename))
            print(f"Saved {len(list_of_dfs)} files in {ioutputdir}")
    
    return out