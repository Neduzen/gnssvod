import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from gnssvod.analysis.vod_timeseries_helper_functions import index_data_files, timePeriod, VOD_base_calc, vod_timeseries_baseline_correction, filter_index_files
import os.path


# CREATE TIME SERIES FROM VOD DATA #
# Use the code from the example scripts provided by Vincent and adjusted to our processing
def calc_timeseries(vod_path, year, baseline_days, out_path_baseline, out_path_timeseries):
    """
    Takes the vod files and creates baseline files as correction
    and construct a time series based on the vod data and baseline.

    Parameters
    ----------
    vod_path: Path were the vod raw files are stored.

    year: Year to be processed as timeseries.

    baseline_days: Amount of days used to calculate the baseline.

    out_path_baseline: Path to store the baseline data.

    out_path_timeseries: Path to store the times series data.
    """
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

    print(f"Time series for {station_name} from {startDate} to {endDate}")
    print(f"Read files at {vod_path}, length: {len(file_dates)}")

    # Generate a baseline file and save it
    bl_name = f'{station_name}_{year}_vod_baseline_{str(baseline_days)}days'
    baseline_param = (baseline_days - 1) / 2
    # VOD_base_calc(file_dates, timeperiod, baseline_param, out_path_baseline, bl_name, save_bs=True)

    # Calculate time series from VOD raw and baseline correction
    baseline_filepath = os.path.join(out_path_baseline, bl_name+'.nc')
    ds_bl = xr.open_dataset(baseline_filepath)
    bl = ds_bl.to_dataframe().dropna(how='all').reorder_levels(['Date', 'CellID']).sort_index()
    ts_name = f'{station_name}_{year}_VOD_timeseries_bl{str(baseline_days)}days'
    vod_timeseries_baseline_correction(file_dates, timeperiod, bl, out_path_timeseries, ts_name, save_ts=True)
    print("Finished VOD timeseries calculations")


def calc_product(timeseries_path, baseline, hour_frequency, out_product, rain_file=None):
    station_name = list(timeseries_path.keys())[0]
    timeseries_path = timeseries_path[station_name]
    out_product = out_product[station_name]
    h_freq = str(hour_frequency)

    file_dates = index_data_files(timeseries_path, station_name, "annual")
    file_dates = filter_index_files(file_dates, baseline=baseline, notincluded="old")

    timeseries_df_years = []
    for f in file_dates["File"]:
        ts_vod = xr.open_dataset(f)
        df_ts_vod = ts_vod.to_dataframe().dropna(how='all').reorder_levels(['Epoch', 'SV']).sort_index()
        hour_sample = df_ts_vod[['VOD_raw', 'VOD', 'VOD_anom', 'CellID']].dropna().groupby(
            [pd.Grouper(freq=f'{h_freq}h', level='Epoch'), 'CellID'])
        count = hour_sample["VOD"].count()
        vod_cell_time_resample = hour_sample.mean()
        vod_cell_time_resample["Count"] = count
        vod_time_resample = vod_cell_time_resample.groupby(["Epoch"]).mean()
        vod_time_resample["Cell_count"] = vod_cell_time_resample["VOD"].groupby(["Epoch"]).count()
        vod_time_resample["Count"] = vod_cell_time_resample["Count"].groupby(["Epoch"]).sum()
        vod_time_resample["VOD_raw_tot_mean"] = df_ts_vod.groupby(pd.Grouper(freq=f'{h_freq}h', level='Epoch')).mean()["VOD_raw"]
        # ts_mean = df_ts_vod.groupby(pd.Grouper(freq=f'{h_freq}h', level='Epoch')).mean()
        timeseries_df_years.append(vod_time_resample)
        # df_ts_vod2 = ts_vod.to_dataframe().dropna(how='all').groupby(['Epoch', 'CellID'])
        # df_ts_vod2 = ts_vod.to_dataframe().dropna(how='all')
        # df_ts_vod3 = df_ts_vod2.groupby([pd.Grouper(freq=f'24h', level='Epoch'), 'CellID']).mean()
        # df_ts_vod4 = df_ts_vod3.groupby(["Epoch"]).mean()
        # df_ts_vod5 = df_ts_vod.groupby(pd.Grouper(freq=f'24h', level='Epoch')).mean()

    df_ts = pd.concat(timeseries_df_years)

    if rain_file is not None:
        calc_rain_product(rain_file, df_ts, baseline, hour_frequency)

    # Calculate the mean instantaneous VOD of a desired length (e.g. baseline length 15 days)
    window = df_ts['VOD_raw'].rolling(window=int(24 * int(baseline) * (1/hour_frequency)), min_periods=1, center=True)
    df_ts['VOD_anom_corr'] = df_ts['VOD_anom'] + window.mean()
    # df_ts = df_ts.rename(columns={"VOD": "VOD_raw", "VOD_anom_corr": "VOD", "VOD_mean": "VOD_bl"})

    # Store the VOD-product file as a .nc
    ds = xr.Dataset.from_dataframe(df_ts)
    ds.Epoch.data[:] = ds.Epoch.data + pd.Timedelta(hours=hour_frequency/2)

    comp = dict(zlib=True, complevel=5)
    encoding = {var: comp for var in ds.data_vars}
    ts_name = fr"{station_name}_VOD_product_bl{baseline}_{h_freq}h"
    filepath = os.path.join(out_product, ts_name + '.nc')
    print(f'Writing the VOD timeseries file to {filepath}')
    ds.to_netcdf(filepath, format="NETCDF4", engine="netcdf4", encoding=encoding)


def calc_rain_product(rain_file, df_ts, baseline, hour_frequency):
    # prc = pd.read_csv(
    #     r"S:\group\rsws\Data\Sites\CH-LAE_Laegeren\ETH_tower_measurements\NewPlatform_2021-2024\precip_cummulative_mm.csv")
    # prc["Epoch"] = pd.to_datetime(prc["Time"], format='%Y-%m-%d %H:%M:%S')
    # prc = prc[["Epoch", "CH-LAE"]]
    # prc_hourly = prc.groupby(pd.Grouper(freq='1h', key='Epoch')).mean()
    # prc_hourly.to_csv(r'S:\group\rsws_gnss\Meteo_data\Laeg\precip_cummulative_mm_hourly.csv', index=True)
    # prc = pd.read_csv(r"S:\group\rsws_gnss\Meteo_data\Laeg\\precip_cummulative_mm_hourly.csv")
    # prc["Epoch"] = pd.to_datetime(prc["Epoch"], format='%Y-%m-%d %H:%M:%S')
    # prc = prc[["Epoch", "CH-LAE"]]
    # # prc = prc[prc["CH-LAE"] > 0]
    # prc['prec'] = prc['CH-LAE'].fillna(method='ffill').diff()
    # # prc.loc[prc['prec'].isnull(), 'prec'] = prc['CH-LAE']
    # prc = prc[["Epoch", "prec"]]
    # # prc_hourly = prc.groupby(pd.Grouper(freq='1h', key='Epoch')).sum()
    # prc.to_csv(r'S:\group\rsws_gnss\Meteo_data\Laeg\Laeg_precip.csv', index=True)

    rain = pd.read_csv(r'S:\group\rsws_gnss\Meteo_data\Laeg\Laeg_precip.csv')
    rain['Epoch'] = pd.to_datetime(rain['Epoch'], format='%Y-%m-%d %H:%M:%S')
    rain = rain.set_index(rain["Epoch"], drop=True)
    vod_rain = pd.merge(df_ts, rain, how="left", left_index=True, right_index=True)  # , on=["Epoch", "date"])
    vod_rain = vod_rain.copy()
    # Set rain flag on hourly data
    vod_rain["Rainflag"] = 0
    vod_rain.loc[vod_rain['prec'] > 0, 'Rainflag'] = -1  # Small rain
    vod_rain.loc[vod_rain['prec'] > 2, 'Rainflag'] = -2  # Large rain
    vod_rain.loc[
        (vod_rain['Rainflag'].shift(+1) < 0) & (vod_rain['Rainflag'] == 0), 'Rainflag'] = -3  # influence of rain
    vod_rain.loc[(vod_rain['Rainflag'].shift(+2) == -2) & (vod_rain['Rainflag'] == 0), 'Rainflag'] = -3
    vod_rain.loc[(vod_rain['Rainflag'].shift(+3) == -2) & (vod_rain['Rainflag'] == 0), 'Rainflag'] = -3
    vod_rain.loc[(vod_rain['Rainflag'].shift(+2) == -1) & (vod_rain['Rainflag'] == 0), 'Rainflag'] = -3
    vod_rain.loc[(vod_rain['Rainflag'].shift(+4) == -2) & (vod_rain['Rainflag'] == 0), 'Rainflag'] = -3
    vod_rain.loc[(vod_rain['Rainflag'].shift(+5) == -2) & (vod_rain['Rainflag'] == 0), 'Rainflag'] = -3

    vod_rain["Cell_count2"] = vod_rain['Cell_count'].copy()
    vod_rain.loc[vod_rain['Rainflag'] < 0, 'Cell_count2'] = 0
    vod_rain["VOD2"] = vod_rain['VOD'].copy()
    vod_rain.loc[vod_rain['Rainflag'] < 0, 'VOD2'] = np.NaN
    vod_rain["VOD_raw2"] = vod_rain['VOD_raw'].copy()
    vod_rain.loc[vod_rain['Rainflag'] < 0, 'VOD_raw2'] = np.NaN
    vod_rain["VOD_anom2"] = vod_rain['VOD_anom'].copy()
    vod_rain.loc[vod_rain['Rainflag'] < 0, 'VOD_anom2'] = np.NaN

    window = vod_rain['VOD_raw2'].rolling(window=int(24 * int(baseline) * (1 / hour_frequency)), min_periods=1,
                                          center=True)
    vod_rain['VOD_anom_corr2'] = vod_rain['VOD_anom2'] + window.mean()
    vod_rain.loc[vod_rain['Rainflag'] < 0, 'VOD_anom_corr2'] = np.NaN

    window = vod_rain['VOD_raw'].rolling(window=int(24 * int(baseline) * (1 / hour_frequency)), min_periods=1,
                                         center=True)
    vod_rain['VOD_anom_corr'] = vod_rain['VOD_anom'] + window.mean()

    import plotly.graph_objects as go
    site = "Laeg"
    print(f"Full timeseries plot of {site}")
    title = f'GNSS VOD timeseries at {site}'
    file_name = f'{site}_vod_timeseries_all_avg.html'
    high_freq = '8H'
    low_freq = '7D'
    tick_format = '%y-%b-%d'
    vod_high_freq = vod_rain.groupby(pd.Grouper(freq=high_freq, level='Epoch')).mean()

    vod_high_freq.loc[vod_high_freq['Cell_count2'] < 370, 'VOD_anom_corr2'] = np.NaN
    vod_high_freq.loc[vod_high_freq['Cell_count2'] < 370, 'VOD2'] = np.NaN
    vod_high_freq["prec"] = vod_high_freq["prec"] * 8
    vod_high_freq["VODdif"] = vod_high_freq["VOD"] - vod_high_freq["VOD_anom_corr"]
    vod_high_freq["VODdif2"] = vod_high_freq["VOD2"] - vod_high_freq["VOD_anom_corr2"]
    vod_high_freq2 = vod_high_freq.groupby(pd.Grouper(freq="1D", level='Epoch')).mean()

    # vod_low_freq = vod_high_freq.rolling('7D', center=True).mean()
    # Figure of the time series with and without baseline
    vod_names = ['VOD_raw', 'VOD', 'VOD2', 'VOD_anom_corr', 'VOD_anom_corr2', "prec"]
    vod_names = ['VOD', 'VOD2', "prec"]
    vod_names = ['VODdif', 'VODdif2', "prec"]

    fig = go.Figure()
    # Add two 4hour means
    i = 0
    col_4 = ["lightgrey", "lightgreen", "red", "orange", "black", "blue"]
    col = ["grey", "darkgreen"]
    for iname in vod_names:
        vodtest = vod_high_freq2.copy()
        vodtest = vodtest[vodtest[iname] >= -0.5]
        fig.add_trace(go.Scatter(
            x=vodtest.index,
            y=vodtest[iname],
            mode='lines+markers',
            name=f'{iname} {high_freq}',
            line=dict(color=col_4[i])
        ))
        i = i + 1
    # Add two 1day means
    # i = 0
    # for iname in vod_names:
    #     fig.add_trace(go.Scatter(
    #         x=vod_low_freq.index,
    #         y=vod_low_freq[iname],
    #         mode='lines',
    #         name=f'{iname} {low_freq}',
    #         line=dict(color=col[i])
    #     ))
    #     i = i + 1

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='GNSS-VOD (L1)',
        xaxis=dict(
            tickformat=tick_format
        ),
        legend_title='Measurements',
        template='plotly_white'
    )
    fig.show()

