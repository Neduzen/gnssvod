## CREATE HEMISPHERIC PLOTS FROM VOD DATA ##
# Use the code from the example scripts provided by Vincent and adjusted to our processing
import os.path

# ---------------
# Set up script
# ---------------
import gnssvod as gv
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import matplotlib.dates as mdates
import pickle as pkl
import plotly.graph_objects as go
from gnssvod.analysis.vod_timeseries_helper_functions import index_data_files, filter_index_files


def monthly_hemi_plot(vod, site, year, month, out_path, is_baseline=False, do_anomaly=False):
    """
    Creates the monthly VOD hemispherical plot for a specific month and year and site.    
    """
    print(f"Hemisphere plot {site} and {year}-{month}")
    vod_name = "VOD"
    file_name = f'{site}_vod_hemplot_{year}-{month:02d}_avg.png'
    z_lim = [-0.1, 3]
    if do_anomaly:
        vod_name = "VOD_anom"
        file_name = f'{site}_vod_anomaly_hemplot_{year}-{month:02d}.png'
        z_lim = [-0.5, 0.5]

    # # Initialize hemispheric grid and patches
    hemi = gv.hemibuild(2)
    patches = hemi.patches()

    if is_baseline:
        vod_avg = (vod[vod.index.get_level_values('Date').month == month].
                   rename(columns={"VOD_mean": "VOD", "VOD_count": "bl_VOD_count"}))
    else:
        vod_avg = vod[vod.index.get_level_values('Epoch').month == month]
        vod_avg = vod_avg[vod_avg.index.get_level_values('Epoch').year == int(year)]

    vod_avg = vod_avg.groupby(['CellID']).mean(['mean', 'std', 'count'])

    # Make the VOD figure
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))

    # Associate the mean values to the patches, join inner will drop patches with no data, making plotting slightly faster
    ivod_data = vod_avg[vod_name].where(vod_avg[f"bl_VOD_count"] > 20)  # select minimal number of observations
    ipatches = pd.concat([patches, ivod_data], join='inner', axis=1)
    # Plotting with colored patches
    pc = PatchCollection(ipatches.Patches, array=ipatches[vod_name], edgecolor='face', linewidth=1)
    pc.set_clim(z_lim)
    ax.add_collection(pc)
    ax.set_rlim([0, 90])
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title(f'{vod_name} - {year} - {month}', fontsize=20)

    plt.colorbar(pc, ax=ax, location='bottom', shrink=.5, pad=0.05, label='GNSS-VOD')
    plt.show()
    plot_path = os.path.join(out_path, file_name)
    fig.savefig(plot_path, bbox_inches='tight', pad_inches=0.1, dpi=300)



def time_plot(vod, site, year=None, out_path=None):
    """
    Creates the times series VOD plots  for a specif site and year.
    Makes 24h and 4h average for VOD_Raw and VOD (corrected).
    """
    if out_path is None:
        raise ValueError("No out_path defined")
    if year is None:
        print(f"Full timeseries plot of {site}")
        title = f'GNSS VOD timeseries at {site}'
        file_name = f'{site}_vod_timeseries_all_avg.html'
        high_freq = '1D'
        low_freq = '7D'
        tick_format = '%y-%b'
        vod_high_freq = vod.groupby(pd.Grouper(freq=high_freq, level='Epoch')).mean()
        vod_low_freq = vod_high_freq.rolling('7D', center=True).mean()
    else:
        print(f"Timeseries plot {site} and {year}")
        high_freq = '8h'
        low_freq = '2D'
        tick_format = '%d-%b'
        title = f'GNSS VOD at {site} of {year}'
        file_name = f'{site}_vod_timeseries_{year}_avg.html'
        vod = vod[vod.index.year == int(year)]
        vod_high_freq = vod.groupby(pd.Grouper(freq=high_freq, level='Epoch')).mean()
        vod_low_freq = vod_high_freq.rolling('7D', center=True).mean()

    # Figure of the time series with and without baseline
    vod_names = ['VOD_raw', 'VOD_anom_corr']

    fig = go.Figure()
    # Add two 4hour means
    i = 0
    col_4 = ["lightgrey", "lightgreen"]
    col = ["grey", "darkgreen"]
    for iname in vod_names:
        fig.add_trace(go.Scatter(
            x=vod_high_freq.index,
            y=vod_high_freq[iname],
            mode='lines+markers',
            name=f'{iname} {high_freq}',
            line=dict(color=col_4[i])
        ))
        i = i + 1
    # Add two 1day means
    i = 0
    for iname in vod_names:
        fig.add_trace(go.Scatter(
            x=vod_low_freq.index,
            y=vod_low_freq[iname],
            mode='lines',
            name=f'{iname} {low_freq}',
            line=dict(color=col[i])
        ))
        i = i + 1

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

    # Save plot as an interactive HTML file
    plot_path = os.path.join(out_path, file_name)
    fig.write_html(plot_path)


def do_plot(timeseries_path, product_path, baseline_path, year, baseline, out_path):
    """
    Creates all needed plots for a site for a year.
    """
    station_name = list(timeseries_path.keys())[0]
    timeseries_path = timeseries_path[station_name]
    product_path = product_path[station_name]
    baseline_path = baseline_path[station_name]
    out_path = out_path[station_name]

    print(f"Plot gnss site {station_name} at {out_path}")
    
    # Load and filter input files
    files_product = index_data_files(product_path, station_name, "all")
    files_product = filter_index_files(files_product, baseline=baseline, notincluded="old")
    ds_product = xr.open_dataset(files_product["File"][0]).to_dataframe()
    
    # Run times series plot
    time_plot(ds_product, station_name, None, out_path)
    time_plot(ds_product, station_name, year, out_path)

    # Monthly VOD plot
    files_bl = index_data_files(baseline_path, station_name, "annual")
    files_names_bl = filter_index_files(files_bl, year=year, baseline=baseline, notincluded="old")["File"]

    if len(files_names_bl) == 0:
        print(f"Error: No timeseries files found for site {station_name} and year {year}")
    if len(files_names_bl) > 1:
        print(f"Error: More than 1 timeseries files found for site {station_name} and year {year}")
        print(f"Files: {files_names_bl}")
        
    # Load the processed VOD data set
    ds = xr.open_dataset(files_names_bl[0])
    # Sorted by Epoch and satellite (Coordinates of ds). All Data variables of
    df = ds.to_dataframe().dropna(how='all')

    out_path_hem = os.path.join(out_path, "hemplot")
    # Run for each month of the year the hemispherical plot
    for month in range(1, 13):
        monthly_hemi_plot(df, station_name, year, month, out_path_hem, is_baseline=True)

    print("Plots finished")


def do_alltime_plot(product_path, baseline, out_path):
    station_name = list(product_path.keys())[0]
    product_path = product_path[station_name]
    out_path = out_path[station_name]

    print(f"Plot gnss site {station_name} at {out_path}")

    # Load and filter input files
    files_product = index_data_files(product_path, station_name, "all")
    files_product = filter_index_files(files_product, baseline=baseline, notincluded="old")
    ds_product = xr.open_dataset(files_product["File"][0]).to_dataframe()
    # Run times series plot
    time_plot(ds_product, station_name, None, out_path)


def analysis_plot(product_path, baseline, out_path):
    site = list(product_path.keys())[0]
    product_path = product_path[site]
    out_path = out_path[site]

    # Load and filter input files
    files_product = index_data_files(product_path, site, "all")
    files_product = filter_index_files(files_product, baseline=baseline, notincluded="old")
    ds_product = xr.open_dataset(files_product["File"][0]).to_dataframe()

    vod=ds_product

    print(f"Full timeseries plot of {site}")
    title = f'GNSS VOD timeseries at {site}'
    iname="VOD"
    file_name = f'{site}_vod_timeseries_annual_avg.html'
    file_name2 = f'{site}_vod_timeseries_anomaly_avg.html'
    high_freq = '1D'
    low_freq = '14D'
    tick_format = '%y-%b'
    vod_high_freq = vod.groupby(pd.Grouper(freq=high_freq, level='Epoch')).mean()
    vod_low_freq = vod_high_freq.rolling('3D', center=True).mean()
    vod_low_freq["doy"]=vod_low_freq.index.dayofyear
    vod_low_freq["year"]=vod_low_freq.index.year

    vod_low_freq2 = vod_high_freq.rolling('5D', center=True).mean()
    vod_low_freq2["doy"]=vod_low_freq2.index.dayofyear
    vod_low_freq2["year"]=vod_low_freq2.index.year

    vod_mean = vod_high_freq.rolling('14D', center=True).mean()
    vod_mean["doy"]=vod_mean.index.dayofyear
    vod_mean["year"]=vod_mean.index.year
    vod_mean["VOD_mean"]=vod_mean["VOD"]
    vod_mean=vod_mean.groupby(vod_mean["doy"]).mean().reset_index()
    vod_mean["doy"]=vod_mean.index

    vod_low_freq2 = pd.merge(vod_low_freq2, vod_mean[["VOD_mean", "doy"]], how="left", on=["doy", "doy"])


    fig = go.Figure()
    # Add two 1day means
    i = 0
    for yeari in [2021,2022,2023,2024,2025]:
        vodplot=vod_low_freq[vod_low_freq.year==yeari]
        fig.add_trace(go.Scatter(
            x=vodplot.doy,
            y=vodplot[iname],
            mode='lines',
            name=f'{yeari}'
        ))
        i = i + 1
    fig.add_trace(go.Scatter(
        x=vod_mean.index,
        y=vod_mean[iname],
        mode='lines',
        name=f'mean',
        line=dict(color="black")
    ))
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
    plot_path = os.path.join(out_path, file_name)
    fig.write_html(plot_path)

    fig = go.Figure()
    for yeari in [2021,2022,2023,2024,2025]:
        vodplot=vod_low_freq2[vod_low_freq2.year==yeari]
        fig.add_trace(go.Scatter(
            x=vodplot.doy,
            y=vodplot[iname]/vodplot["VOD_mean"],
            mode='lines',
            name=f'{yeari}'
        ))
        i = i + 1
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
    # Save plot as an interactive HTML file
    plot_path = os.path.join(out_path, file_name2)
    fig.write_html(plot_path)

    prc = pd.read_csv(r"S:\group\rsws\Data\Sites\CH-LAE_Laegeren\ETH_tower_measurements\NewPlatform_2021-2024\precip_cummulative_mm.csv")
    prc["date"] = pd.to_datetime(prc["Time"], format='%Y-%m-%d %H:%M:%S')
    prc = prc[["date", "CH-LAE"]]
    prc = prc[prc["CH-LAE"] > 0]
    prc['prec'] = prc['CH-LAE'].fillna(method='ffill').diff()
    prc.loc[prc['prec'].isnull(), 'prec'] = prc['CH-LAE']
    prc_6hours = prc.groupby(pd.Grouper(freq='1H', key='date')).sum()

    # airt = pd.read_csv(
    #     r"S:\group\rsws\Data\Sites\CH-LAE_Laegeren\ETH_tower_measurements\NewPlatform_2021-2024\shortwave_incoming_radiation_Wm-2.csv")
    # airt["date"] = pd.to_datetime(airt["Time"], format='%Y-%m-%d %H:%M:%S')
    # airt = airt[["date", "CH-LAE"]]
    # airt = airt[airt["CH-LAE"] > 0]
    # # vpd_day = vpd[vpd.Time.dt.hour > 8]
    # airt_daily = airt.groupby(pd.Grouper(freq='1D', key='date')).mean()
    # # vpd_daily = vpd_daily.rolling('2D', center=True).mean()
    #
    # # vpd_daily = vpd.rolling('3D', center=True).mean()
    # airt_daily["doy"] = airt_daily.index.dayofyear
    # airt_daily["year"] = airt_daily.index.year
    # airt_daily["vpd_rel"] = 0.8 + ((airt_daily["CH-LAE"] - np.nanmin(airt_daily["CH-LAE"])) / (
    #     np.nanmax(airt_daily["CH-LAE"] - np.nanmin(airt_daily["CH-LAE"])))) / 2
    #
    # vpd = pd.read_csv(
    #     r"S:\group\rsws\Data\Sites\CH-LAE_Laegeren\ETH_tower_measurements\NewPlatform_2021-2024\soil_water_Content_5cm_depth.csv")
    # vpd["date"] = pd.to_datetime(vpd["Time"], format='%Y-%m-%d %H:%M:%S')
    # vpd = vpd[["date", "CH-LAE"]]
    # vpd = vpd[vpd["CH-LAE"] > 0]
    # # vpd_day = vpd[vpd.Time.dt.hour > 8]
    # vpd_daily = vpd.groupby(pd.Grouper(freq='1D', key='date')).mean()
    # # vpd_daily = vpd_daily.rolling('2D', center=True).mean()
    #
    # # vpd_daily = vpd.rolling('3D', center=True).mean()
    # vpd_daily["doy"] = vpd_daily.index.dayofyear
    # vpd_daily["year"] = vpd_daily.index.year
    # vpd_daily["vpd_rel"] = 0.8 + ((vpd_daily["CH-LAE"] - np.nanmin(vpd_daily["CH-LAE"])) / (
    #     np.nanmax(vpd_daily["CH-LAE"] - np.nanmin(vpd_daily["CH-LAE"])))) / 2
    # # vpd_daily["CH-AWS"]/15+0.3
    #
    # vod_low_freq = vod_high_freq.rolling('2D', center=True).mean()
    # vod_low_freq["doy"] = vod_low_freq.index.dayofyear
    # vod_low_freq["year"] = vod_low_freq.index.year
    # fig = go.Figure()
    # # Add two 1day means
    # i = 0
    # for yeari in [2021, 2022, 2023, 2024, 2025]:
    #     vodplot = vod_low_freq[vod_low_freq.year == yeari]
    #     fig.add_trace(go.Scatter(
    #         x=vodplot.doy,
    #         y=vodplot[iname],
    #         mode='lines',
    #         name=f'{yeari}'
    #     ))
    #     i = i + 1
    # fig.add_trace(go.Scatter(
    #     x=vod_mean.index,
    #     y=vod_mean[iname],
    #     mode='lines',
    #     name=f'mean',
    #     line=dict(color="black")
    # ))
    # # Update layout
    # fig.update_layout(
    #     title=title,
    #     xaxis_title='Date',
    #     yaxis_title='GNSS-VOD (L1)',
    #     xaxis=dict(
    #         tickformat=tick_format
    #     ),
    #     legend_title='Measurements',
    #     template='plotly_white'
    # )
    # i = 0
    # for yeari in [2021, 2022, 2023, 2024, 2025]:
    #     vpd_plot = vpd_daily[vpd_daily.year == yeari]
    #     fig.add_trace(go.Scatter(
    #         x=vpd_plot.doy,
    #         y=vpd_plot["vpd_rel"],
    #         mode='lines',
    #         name=f'vpd - {yeari}'
    #     ))
    #     i = i + 1
    # i = 0
    # fig.show()


    # VOD SNR and precipitation
    # prc = pd.read_csv(r"S:\group\rsws\Data\Sites\CH-LAE_Laegeren\ETH_tower_measurements\NewPlatform_2021-2024\precip_cummulative_mm.csv")
    # prc["date"] = pd.to_datetime(prc["Time"], format='%Y-%m-%d %H:%M:%S')
    # prc = prc[["date", "CH-LAE"]]
    # prc = prc[prc["CH-LAE"] > 0]
    # prc['prec'] = prc['CH-LAE'].fillna(method='ffill').diff()
    # prc.loc[prc['prec'].isnull(), 'prec'] = prc['CH-LAE']
    # prc_6hours = prc.groupby(pd.Grouper(freq='1H', key='date')).sum()
    # #
    # #
    # iref2 = data.xs(icase[1][0], level='Station')
    #
    # t2 = iref2[iref2["Elevation"] > 40]
    # # t2=t2[t2.index.get_level_values('SV').isin(['R14', 'R11', 'R15','R16', 'R05', 'R12', 'R07', 'R17', 'R03', 'R02', 'R24', 'R26', 'R21', 'R22', 'R09','R04','G09',"C20", "G08", "G30", "C22", "E18", "C21", 'R18', 'G26', "C35", "G15", "C32", "C24", "G24", "C27"])]
    # t = t2["S2"].groupby(pd.Grouper(freq='1H', level='Epoch')).mean()
    # prc_6hours_t = prc_6hours[prc_6hours.index > np.nanmin(t.index)]
    # prc_6hours_t = prc_6hours_t[prc_6hours_t.index < np.nanmax(t.index)]
    # fig = go.Figure()
    # tsv = t2["S2"].groupby(pd.Grouper(level='SV')).mean()
    #
    # # t2=t2[t2.index.get_level_values('SV').isin(
    # # ['R14', 'R11', 'R15','R16', 'R05', 'R12', 'R07', 'R17', 'R03', 'R02', 'R24', 'R26', 'R21', 'R22', 'R09','R04'])]
    #
    # fig.add_trace(go.Scatter(
    #     x=t.index,
    #     y=t.values,
    #     mode='lines',
    #     name=f'mean',
    #     line=dict(color="black")
    # ))
    # fig.add_trace(go.Scatter(
    #     x=prc_6hours_t.index,
    #     y=prc_6hours_t["prec"],
    #     mode='lines',
    #     name=f'mean',
    #     line=dict(color="blue")
    # ))
    # fig.show()


    # def filter_Rain():
    # vod_high_freq = vod.groupby(pd.Grouper(freq=high_freq, level='Epoch')).mean()
    # vod_12h = vod.groupby(pd.Grouper(freq="12H", level='Epoch')).mean()

    # vod_low_freq = vod_high_freq.rolling('3D', center=True).mean()
    #
    # tick_format = '%y-%b-%d'
    # fig = go.Figure()
    # # for yeari in [2021,2022,2023,2024,2025]:
    # #     vodplot=vod_low_freq2[vod_low_freq2.year==yeari]
    # fig.add_trace(go.Scatter(
    #     x=vod_high_freq.index,
    #     y=vod_high_freq[iname],
    #     mode='lines',
    #     name=f'vod',
    #     line=dict(color="green")
    # ))
    #     # i = i + 1
    #
    # prc = pd.read_csv(
    #     r"S:\group\rsws\Data\Sites\CH-LAE_Laegeren\ETH_tower_measurements\NewPlatform_2021-2024\precip_cummulative_mm.csv")
    # prc["Epoch"] = pd.to_datetime(prc["Time"], format='%Y-%m-%d %H:%M:%S')
    # prc = prc[["Epoch", "CH-LAE"]]
    # prc = prc[prc["CH-LAE"] > 0]
    # prc['prec'] = prc['CH-LAE'].fillna(method='ffill').diff()
    # prc.loc[prc['prec'].isnull(), 'prec'] = prc['CH-LAE']
    # prc_6hours = prc.groupby(pd.Grouper(freq='12h', key='Epoch')).sum()
    #
    # vod_rain = pd.merge(vod_12h, prc_6hours, how="left", left_index=True, right_index=True)  # , on=["Epoch", "date"])
    # vod_rain = vod_rain.copy()
    # vod_rain["VOD2"] = vod_rain['VOD'].copy()
    # vod_rain.loc[vod_rain['prec'] > 0, 'VOD2'] = np.nan
    # vod_rain.loc[vod_rain['prec'] > 3, 'VOD2'] = np.nan  # sets current value to NAN if >3
    # vod_rain.loc[vod_rain['prec'].shift(-1) > 3, 'VOD2'] = np.nan  # shifts one down and sets
    #
    # vod_day = vod_rain.groupby(pd.Grouper(freq="1D", level='Epoch')).mean()
    #
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(
    #     x=vod_rain.index,
    #     y=vod_rain["VOD"],
    #     mode='lines',
    #     name=f'VOD',
    #     line=dict(color="green")
    # ))
    # fig.add_trace(go.Scatter(
    #     x=vod_rain.index,
    #     y=vod_rain["VOD2"],
    #     mode='lines',
    #     name=f'VOD',
    #     line=dict(color="black")
    # ))
    # fig.add_trace(go.Scatter(
    #     x=vod_day.index,
    #     y=vod_day["VOD"],
    #     mode='lines',
    #     name=f'VOD day',
    #     line=dict(color="yellow")
    # ))
    # fig.add_trace(go.Scatter(
    #     x=vod_day.index,
    #     y=vod_day["VOD2"],
    #     mode='lines',
    #     name=f'VOD day',
    #     line=dict(color="grey")
    # ))
    # fig.add_trace(go.Scatter(
    #     x=vod_rain.index,
    #     y=vod_rain["prec"],
    #     mode='lines',
    #     name=f'precip',
    #     line=dict(color="blue")
    # ))
    # # Update layout
    # fig.update_layout(
    #     title=title,
    #     xaxis_title='Date',
    #     yaxis_title='GNSS-VOD (L1)',
    #     xaxis=dict(
    #         tickformat=tick_format
    #     ),
    #     legend_title='Measurements',
    #     template='plotly_white'
    # )
    #
    # fig.show()