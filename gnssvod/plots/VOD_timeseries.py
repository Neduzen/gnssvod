import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from gnssvod.plots.VOD_timeseries_helper_functions import index_data_files, timePeriod, VOD_base_calc, vod_timeseries_baseline_correction
import os.path

# CREATE TIME SERIES FROM VOD DATA #
# Use the code from the example scripts provided by Vincent and adjusted to our processing

def calc_timeseries(vod_path, year, baseline_days, out_path_baseline, out_path_timeseries):
    station_name = list(vod_path.keys())[0]
    vod_path = vod_path[station_name]
    out_path_baseline = out_path_baseline[station_name]
    out_path_timeseries = out_path_timeseries[station_name]

    # Generate a dataframe with all the files available and their start and end date
    file_dates = index_data_files(vod_path, station_name)

    # Generate a timeperiod index to loop through
    startDate = f'{year}-01-01'
    endDate = f'{year}-12-31'
    timeperiod = timePeriod(startDate, endDate)

    # Generate a baseline file and save it
    bl_name = f'{station_name}_vod_baseline_{str(baseline_days)}days_{year}'
    baseline_param = (baseline_days - 1) / 2
    # vod_baseline = VOD_base_calc(file_dates, timeperiod, baseline_param, out_path_baseline, bl_name, save_bs=True)

    # Calculate time series from VOD raw and baseline correction
    baseline_filepath = os.path.join(out_path_baseline, bl_name+'.nc')
    ds_bl = xr.open_dataset(baseline_filepath)
    bl = ds_bl.to_dataframe().dropna(how='all').reorder_levels(['Date', 'CellID']).sort_index()
    ts_name = f'{station_name}_{year}_VOD_timeseries_bl{str(baseline_days)}days_v1'
    vod = vod_timeseries_baseline_correction(file_dates, timeperiod, bl, out_path_timeseries, ts_name, save_ts=True)


def calc_product(baseline_path, baseline, hour_frequency, out_product):
    station_name = list(baseline_path.keys())[0]
    baseline_path = baseline_path[station_name]
    out_product = out_product[station_name]
    h_freq = str(hour_frequency)

    file_dates = index_data_files(baseline_path, station_name, "annual")

    timeseries_df_years = []
    for f in file_dates["File"]:
        ts_vod = xr.open_dataset(f)
        df_ts_vod = ts_vod.to_dataframe().dropna(how='all').reorder_levels(['Epoch', 'SV']).sort_index()
        df_ts_vod = df_ts_vod[["VOD", "VOD_mean", "VOD_anom"]]
        ts_mean = df_ts_vod.groupby(pd.Grouper(freq=f'{h_freq}h', level='Epoch')).mean()
        timeseries_df_years.append(ts_mean)

    df_ts = pd.concat(timeseries_df_years)

    # Calculate the mean instantaneous VOD of a desired length (e.g. baseline length 15 days)
    window = df_ts['VOD'].rolling(window=int(24 * int(baseline) * (1/hour_frequency)), min_periods=1, center=True)
    df_ts['VOD_anom_corr'] = df_ts['VOD_anom'] + window.mean()
    df_ts = df_ts.rename(columns={"VOD": "VOD_raw", "VOD_anom_corr": "VOD", "VOD_mean": "VOD_bl"})

    # Store the VOD_baseline file as a .nc
    ds = xr.Dataset.from_dataframe(df_ts)
    comp = dict(zlib=True, complevel=5)
    encoding = {var: comp for var in ds.data_vars}

    ts_name = fr"{station_name}_VOD_product_bl{baseline}_{h_freq}h"
    # Save as NetCDF file
    filepath = os.path.join(out_product, ts_name + '.nc')
    print(f'Writing the VOD timeseries file to {filepath}')
    ds.to_netcdf(filepath, format="NETCDF4", engine="netcdf4", encoding=encoding)



