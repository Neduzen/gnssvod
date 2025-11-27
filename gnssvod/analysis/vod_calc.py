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

def process_vod(filepattern,pairings,bands,timeinterval, outputdir):
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
    station_name = list(filepattern.keys())[0]
    filepattern = filepattern[station_name]
    files = get_filelist({station_name: filepattern})
    success = 0
    total = len(timeinterval)

    # Loop through interval and calc and store VOD per interval
    for i in range(0, total):
        # Get current interval and filter files
        print(f"VOD calculation of month:")
        sub_interval = pd.interval_range(start=timeinterval[i].left,
                                         end=timeinterval[i].right, freq='MS')
        files_for_month = filter_filelist(files[station_name], sub_interval, [station_name + "_"])
        if len(files_for_month) > 0:  # check if the list is empty
            out = calculate_vod(files_for_month, sub_interval, pairings, bands, outputdir, station_name)
            if out == 0:
                success += 1
        else:
            print("No data for interval.")
    print(f"VOD calculation complete. {success} of {total} processed and saved.")



def calculate_vod(files, timeinterval, pairings, bands, outputdir, station_name):
    """Processes data for the given time interval."""
    vod_var_name = "VOD"
    # Open paired files
    data = xr.open_mfdataset(files).to_dataframe().dropna(how='all')

    # Calculate VOD based on pairings
    for icase in pairings.items():
        iref = data.xs(icase[1][0], level='Station')
        igrn = data.xs(icase[1][1], level='Station')
        idat = iref.merge(igrn, on=['Epoch', 'SV'], suffixes=['_ref', '_grn'])
        for ivod in bands.items():
            ivars = np.intersect1d(data.columns.to_list(), ivod[1])
            for ivar in ivars:
                irefname = f"{ivar}_ref"
                igrnname = f"{ivar}_grn"
                ielename = f"Elevation_grn"
                # VOD calculation formula
                idat[ivar] = -np.log(np.power(10, (idat[igrnname] - idat[irefname]) / 10)) \
                                     * np.cos(np.deg2rad(90 - idat[ielename]))

        # idat[vod_var_name] = np.nan
        # for ivar in ivars:
        #     idat[vod_var_name] = idat[vod_var_name].fillna(idat[ivar].copy())

        # TODO: IMPROVE: Filter quality
        band1 = ivars[0]
        band2 = ivars[1]
        irefname = f"{band1}_ref"
        igrnname = f"{band1}_grn"
        len_before = len(idat[band1])
        # Filter low signal data
        idat[band1][(idat[irefname] < 25) | (idat[igrnname] < 5)] = np.NaN
        idat[band2][(idat[f"{band2}_ref"] < 25) | (idat[f"{band2}_grn"] < 5)] = np.NaN
        # Filter both NaN vod data
        idat = idat[~idat[band1].isna() | ~idat[band2].isna()]
        print(f"Removed {len_before - len(idat[band1])} low signal observations.")

        idat = idat[list([irefname, igrnname] + bands[station_name]) + ['Azimuth_ref', 'Elevation_ref']].rename(
            columns={band1: 'VOD', band2: f'VOD_{band2}', 'Azimuth_ref': 'Azimuth', 'Elevation_ref': 'Elevation'})

    # Save vod file
    return save_vod_data(idat, timeinterval, station_name, outputdir)


def save_vod_data(idat, timeinterval, station_name, outputdir):
    # Save to NetCDF files of time interval
    if outputdir:
        date_s = timeinterval[0].left.strftime("%Y%m%d")
        date_e = (timeinterval[0].right - datetime.timedelta(milliseconds=1)).strftime("%Y%m%d")
        filename = f'vod_{station_name}_{date_s}_{date_e}.nc'  # Monthly filename

        outputdir = outputdir[station_name]
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)

        if not idat.empty:  # Check if DataFrame is empty
            ds = idat.to_xarray()
            ds.to_netcdf(os.path.join(outputdir, filename))
            print(f"Saved monthly data ({len(idat)} obs) to {filename}")
            return 0
        else:
            print(f"No data for {date_s} to {date_e}, no file saved")
            return -1
    else:
        print("Data is not saved as defined in setup")
        return -1
