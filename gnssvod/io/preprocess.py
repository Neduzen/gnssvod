"""
preprocess reads files and returns analysis-ready DataSet

pair_obs merges and pairs observations from sites according to specified pairing rules over the desired time intervals
"""
# ===========================================================
# ========================= imports =========================
import os
import sys
from tqdm import trange, tqdm
import time
import glob
import datetime
import numpy as np
import pandas as pd
import xarray as xr
from gnssvod.io.readFile import read_obsFile, FileError
from gnssvod.funcs.checkif import (isfloat, isint, isexist)
from gnssvod.funcs.date import doy2date
from gnssvod.position.interpolation import sp3_interp_fast
from gnssvod.position.position import gnssDataframe
from gnssvod.funcs.constants import _system_name
import pdb
import logging

# ===========================================================

#-------------------------------------------------------------------------
#----------------- FILE SELECTION AND BATCH PROCESSING -------------------
#-------------------------------------------------------------------------
def preprocess(filepattern,
               orbit=True,
               interval=None,
               keepvars=None,
               outputdir=None,
               overwrite=False,
               unzip_path=None,
               outputresult=False,
               time_period=None,
               splitter=["_raw"],
               rename_vars=[]):
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
        
    outputdir: dictionary (optional)
        A dictionary of station names and folders indicating where to save the preprocessed data
        For example outputdir={'station1':'/path/where/to/save/preprocessed/data'}
        Dictionary keys must be the same as in the filepattern argument
        Data will be saved as a netcdf file, recycling the original file name
        If this argument is None, data won't be saved

    overwrite: bool (optional)
        If False (default), RINEX files with an existing matching files in the 
        specified output directory will be skipped entirely

    outputresult: bool (optional)
        If True, observation objects will also be returned as a dictionary
        
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

        # filter files by time period
        if time_period is not None:
            filelist = filter_filelist(filelist, time_period, splitter=splitter)

        # checking which files will be skipped (if necessary)
        if (not overwrite) and (outputdir is not None):
            # gather all files that already exist in the outputdir
            extension = f"/*.nc"  # Linux extension
            if os.name == 'nt':  # 'nt' represents Windows
                extension = f"\\*.nc"
            files_to_skip = get_filelist({station_name: outputdir[station_name]+extension})
            files_to_skip = [os.path.basename(x) for x in files_to_skip[station_name]]
        else:
            files_to_skip = []
        
        # for each file
        result = []
        for i, filename in tqdm(enumerate(filelist), desc="Preprocessing"):#, file=sys.stdout, colour='GREEN'):#, unit="iteration", position=0, leave=True):
            try:
                # determine the name of the output file that will be saved at the end of the loop
                out_name = os.path.splitext(os.path.basename(filename))[0]+'.nc'
                # if the name of the saved output file is in the files to skip, skip processing
                if out_name in files_to_skip:
                    print(f"{out_name} already exists, skipping.. (pass overwrite=True to overwrite)")
                    continue # skip remainder of loop and go directly to next filename

                # read in the file
                x = read_obsFile(filename, unzip_path=unzip_path)
                if x is not None:
                    print(f"Processing {len(x.observation):n} individual observations")

                    # Rename columns
                    for rc in rename_vars:
                        x.observation = x.observation.rename(
                            columns={rc[0]: rc[1]})

                    # only keep required vars
                    if keepvars is not None:
                        # only keep rows for which required vars are not NA
                        x.observation = x.observation.dropna(how='all', subset=keepvars)
                        # subselect only the required vars, + always keep 'epoch' and 'SYSTEM'
                        x.observation_types = np.unique(np.concatenate((keepvars, ['epoch', 'SYSTEM'])))
                        x.observation = x.observation[x.observation_types]

                    if len(x.observation.epoch) == 0:
                        raise FileError(f'No observations for the rinex file {filename}.')

                    # resample if required
                    if interval is not None:
                        x = resample_obs(x,interval)

                    # calculate Azimuth and Elevation if required
                    if orbit:
                        print(f"Calculating Azimuth and Elevation")
                        # note: orbit cannot be parallelized easily because it
                        # downloads and unzips third-party files in the current directory
                        if not 'orbit_data' in locals():
                            # if there is no previous orbit data, the orbit data is returned as well
                            x, orbit_data = add_azi_ele(x)
                        else:
                            # on following iterations the orbit data is tentatively recycled to reduce computational time
                            x, orbit_data = add_azi_ele(x, orbit_data)

                    # make sure we drop any duplicates
                    x.observation = x.observation[~x.observation.index.duplicated(keep='first')]

                    # store result in memory
                    if outputresult:
                        result[i] = x

                    # write to file if required
                    if outputdir is not None:
                        ioutputdir = outputdir[station_name]
                        # check that the output directory exists
                        if not os.path.exists(ioutputdir):
                            os.makedirs(ioutputdir)
                        # delete file if it exists
                        out_path = os.path.join(ioutputdir, out_name)
                        if os.path.exists(out_path):
                            os.remove(out_path)
                        # save as NetCDF
                        ds = x.observation.to_xarray()
                        ds.attrs['filename'] = x.filename
                        ds.attrs['observation_types'] = x.observation_types
                        ds.attrs['epoch'] = x.epoch.isoformat()
                        ds.attrs['approx_position'] = x.approx_position
                        ds.to_netcdf(out_path)
                        print(f"Saved {len(x.observation):n} individual observations in {out_name}")
                else:
                    print(f"File was empty or not there: {filename}")
            except FileError as ex:
                logging.error(f"File error for {out_name} while execution and not successfully processed: {ex}")
                print(f"Warning for file {out_name} while execution and not successfully processed: {ex}")
        # store station in memory if required
        if outputresult:
            out[station_name]=result
        logging.info(f"From {len(item[1])} filtered to {len(filelist)} saved into .nc files")

    if outputresult:
        return out
    else:
        return

def resample_obs(obs,interval):
    obs.observation['SYSTEM'] = pd.NA
    obs.observation = obs.observation.groupby([pd.Grouper(freq=interval, level='Epoch'),pd.Grouper(level='SV')]).mean()
    obs.observation['epoch'] = obs.observation.index.get_level_values('Epoch')
    obs.observation['SYSTEM'] = _system_name(obs.observation.index.get_level_values("SV"))
    obs.interval = pd.Timedelta(interval).seconds
    return obs

def add_azi_ele(obs, orbit_data=None):
    if orbit_data is None:
        do = True
    elif (orbit_data.my_epoch == obs.epoch) and (orbit_data.my_interval == obs.interval):
        # if the orbit for the day corresponding to the epoch and interval is the same as the one that was passed, just reuse it. This drastically reduces the number of times orbit files have to be read and interpolated.
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
            print(f"no files with .nc with {search_pattern}")
            # raise Warning(f"Could not find any files matching the pattern {search_pattern}")
        filelists[station_name] = flist
    return filelists


def filter_filelist(files, time_period, splitter=["raw_"]):
    date_min = time_period[0].left
    date_max = time_period[-1].right
    print(f"Filter files between {date_min} and {date_max}")

    filtered = []
    for f in files:
        try:
            date_str = None
            for s in splitter:
                if s in f:
                    date_str = f.split(s)[1][:10]
                else:
                    continue
            date_time = pd.to_datetime(date_str, format='%Y%m%d%H')
            if date_min <= date_time < date_max:
                filtered.append(f)
        except:
            continue
    print(f"From {len(files)} filtered to {len(filtered)}")

    return filtered

#--------------------------------------------------------------------------
#----------------- PAIRING OBSERVATION FILES FROM SITES -------------------
#-------------------------------------------------------------------------- 

def pair_obs(filepattern, pairings, timeintervals, keepvars=None, outputdir=None, time_period=None):
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
        overall_interval = pd.Interval(left=timeintervals.min().left, right=timeintervals.max().right)
        print(f"Interval: {overall_interval}")
        # get all files
        ref_files = get_filelist({ref_name:filepattern[ref_name]})
        grn_files = get_filelist({grn_name:filepattern[grn_name]})
        # # filter files by time period
        if time_period is not None:
            ref_files[ref_name] = filter_filelist(ref_files[ref_name], time_period)
            grn_files[grn_name] = filter_filelist(grn_files[grn_name], time_period)

        # get Epochs from all files
        ref_epochs = [xr.open_mfdataset(x).Epoch for x in ref_files[ref_name]]
        grn_epochs = [xr.open_mfdataset(x).Epoch for x in grn_files[grn_name]]
        # check which files have data that overlaps with the desired time intervals
        ref_isin = [overall_interval.overlaps(pd.Interval(left=pd.Timestamp(x.values.min()),
                                                          right=pd.Timestamp(x.values.max()))) for x in ref_epochs]
        grn_isin = [overall_interval.overlaps(pd.Interval(left=pd.Timestamp(x.values.min()),
                                                          right=pd.Timestamp(x.values.max()))) for x in grn_epochs]
        print(f'Found {sum(ref_isin)} files for {ref_name} and {sum(grn_isin)} for {grn_name}')
        print(f'Reading')
        # open those files and convert them to pandas dataframes
        ref_data = [xr.open_mfdataset(x).to_dataframe().dropna(how='all',subset=['epoch']) \
                    for x in np.array(ref_files[ref_name])[ref_isin]]
        grn_data = [xr.open_mfdataset(x).to_dataframe().dropna(how='all',subset=['epoch']) \
                    for x in np.array(grn_files[grn_name])[grn_isin]]
        # concatenate, drop duplicates and sort the dataframes
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
        out[case_name] = [x for x in iout.groupby(pd.cut(iout.index.get_level_values('Epoch').tolist(), time_period))]
        
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
                # convert dataframe to xarray for saving to netcdf (if df is not empty)
                if len(df[1])>0:
                    ds = df[1].to_xarray()
                    ds.to_netcdf(os.path.join(ioutputdir,filename))
                    print(f"Saved {len(df[1])} obs in {filename}")
                else:
                    print(f"No data for timestep {ts}, no file saved")
    return out

#--------------------------------------------------------------------------
#----------------- CALCULATING VOD -------------------
#-------------------------------------------------------------------------- 

def calc_vod(filepattern, pairings, outputdir=None, time_period=None):
    """
    Combines a list of NetCDF files containing paired GNSS receiver data, calculates VOD and returns that data.

    The paired GNSS receiver data is typically generated with the function 'pair_obs'.
    
    VOD is calculated based on custom pairing rules indicating the input variables that need to be used.
    
    Parameters
    ----------
    filepattern: dictionary 
        Dictionary of case names and UNIX-style patterns to find the processed NetCDF files.
        For example filepattern={'case1':'/path/to/files/of/case1/*.nc',
                                 'case2':'/path/to/files/of/case2/*.nc'}
    
    pairings: dictionary 
        Dictionary of names associated to a tuple of three variables names indicating what variables to use to calculate VOD, with the reference station given first, the subcanopy station second, and the elevation third.
        For example pairings={'VOD1':('S1C_ref','S1C_grn','Elevation_grn'),
                              'VOD2':('S2C_ref','S2C_grn','Elevation_grn')}
        
    Returns
    -------
    Dictionary of case names associated with dataframes containing the output for each case
    
    """
    out=dict()
    for item in filepattern.items():
        case_name = item[0]
        print(f'Processing {case_name}')
        files = get_filelist({case_name:filepattern[case_name]})
        # Overall interval
        end_datetime = time_period.max().right-datetime.timedelta(microseconds=1)
        overall_interval = pd.Interval(left=time_period.min().left, right=end_datetime)
        print(f"Interval: {overall_interval}")
        # # filter files by time period
        if time_period is not None:
            files[case_name] = filter_filelist(files[case_name], time_period, splitter=[f'{case_name}_'])
        # read in all data
        data = [xr.open_mfdataset(x).to_dataframe().dropna(how='all') for x in files[case_name]]
        # concatenate
        data = pd.concat(data)
        # calculate VOD based on pairings
        for ivod in pairings.items():
            varname_vod = ivod[0]
            varname_ref = ivod[1][0]
            varname_grn = ivod[1][1]
            varname_ele = ivod[1][2]
            data["VOD"] = -np.log(np.power(10,(data[varname_grn]-data[varname_ref])/10)) \
                                * np.cos(np.deg2rad(90-data[varname_ele]))
        # store result in dictionary
        # out[varname_vod] = data

        # Only keep rows with VOD
        keep_vars = ["VOD", "epoch_ref", "Azimuth_ref", "Elevation_ref", "SYSTEM_ref"]
        if keep_vars is not None:
            data = data[keep_vars].dropna(how='all')
        data = data.dropna(subset=['VOD'])

        # split the dataframe into multiple dataframes according to timeintervals
        out[case_name] = [x for x in data.groupby(pd.cut(data.index.get_level_values('Epoch').tolist(), time_period, right=False))]

        if outputdir:
            outputdir = outputdir[case_name]
            for item in out.items():
                case_name = item[0]
                list_of_dfs = item[1]
                if not os.path.exists(outputdir):
                    os.makedirs(outputdir)
                print(f'Saving files for {case_name} in {outputdir}')
                for df in list_of_dfs:
                    # date_s = np.min(df[1].axes[0])[0].strftime("%Y%m%d")
                    # date_e = np.max(df[1].axes[0])[0].strftime("%Y%m%d")
                    date_s = df[0].left.strftime("%Y%m%d")
                    date_e = (df[0].right - datetime.timedelta(milliseconds=1)).strftime("%Y%m%d")
                    filename = f'vod_{case_name}_{date_s}_{date_e}.nc'
                    # convert dataframe to xarray for saving to netcdf (if df is not empty)
                    if len(df[1]) > 0:
                        ds = df[1].to_xarray()
                        ds.to_netcdf(os.path.join(outputdir, filename))
                        print(f"Saved {len(df[1])} obs in {filename}")
                    else:
                        print(f"No databetween {date_s} and {date_e}, no file saved")
                print(f"Saved vods' to {outputdir}")
    return out