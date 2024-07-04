"""
calc_vod calculates VOD according to specified pairing rules
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
import warnings
from gnssvod.io.preprocess import get_filelist, filter_filelist
import pdb
#--------------------------------------------------------------------------
#----------------- CALCULATING VOD -------------------
#-------------------------------------------------------------------------- 

def calc_vod(filepattern,pairings,bands,timeinterval, outputdir):
    """
    Combines a list of NetCDF files containing gathered GNSS receiver data, calculates VOD and returns that data.
    
    The gathered GNSS receiver data is typically generated with the function 'gather_stations'.
    
    VOD is calculated based on pairing rules referring to station names.
    
    Parameters
    ----------
    filepattern: dictionary 
        a UNIX-style pattern to find the processed NetCDF files.
        For example filepattern='/path/to/files/of/case1/*.nc'
    
    pairings: dictionary
        A dictionary of pairs of station names indicating first the reference station and second the ground station.
        For example pairings={'Laeg1':('Laeg2_Twr','Laeg1_Grnd')}

    bands: dictionary
        Dictionary of column names to be used for combining different bands
        For example bands={'VOD_L1':['S1','S1X','S1C']}
        
    Returns
    -------
    Dictionary of case names associated with dataframes containing the output for each case
    
    """
    vod_var_name = "VOD"
    station_name = list(filepattern.keys())[0]
    filepattern = filepattern[station_name]
    files = get_filelist({station_name: filepattern})
    # Filter files on overall interval
    end_datetime = timeinterval.max().right - datetime.timedelta(microseconds=1)
    overall_interval = pd.Interval(left=timeinterval.min().left, right=end_datetime)
    print(f"Interval: {overall_interval}")
    filelist = filter_filelist(files[station_name], timeinterval, [station_name+"_"])

    # read in all data
    data = [xr.open_mfdataset(x).to_dataframe().dropna(how='all') for x in files[station_name]]
    # concatenate
    data = pd.concat(data)
    # calculate VOD based on pairings
    out = dict()
    for icase in pairings.items():
        iref = data.xs(icase[1][0],level='Station')
        igrn = data.xs(icase[1][1],level='Station')
        idat = iref.merge(igrn,on=['Epoch','SV'],suffixes=['_ref','_grn'])
        for ivod in bands.items():
            ivars = np.intersect1d(data.columns.to_list(),ivod[1])
            for ivar in ivars:
                irefname = f"{ivar}_ref"
                igrnname = f"{ivar}_grn"
                ielename = f"Elevation_grn"
                idat[ivar] = -np.log(np.power(10,(idat[igrnname]-idat[irefname])/10)) \
                            *np.cos(np.deg2rad(90-idat[ielename]))
            
            idat[vod_var_name] = np.nan
            for ivar in ivars:
                idat[vod_var_name] = idat[vod_var_name].fillna(idat[ivar].copy())

        idat = idat[list([vod_var_name] + bands[station_name]) + ['Azimuth_ref', 'Elevation_ref']].rename(
    columns={'Azimuth_ref': 'Azimuth', 'Elevation_ref': 'Elevation'})
        # store result in dictionary
        out[station_name]=idat

        # split the dataframe into multiple dataframes according to time intervals
        out[station_name] = [x for x in
                             idat.groupby(
                                 pd.cut(idat.index.get_level_values('Epoch').tolist(), timeinterval, right=False))]
        # Save according to time interval
        if outputdir:
            outputdir = outputdir[station_name]
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