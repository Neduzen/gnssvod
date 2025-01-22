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
from gnssvod.plots.VOD_timeseries_helper_functions import index_data_files, filter_index_files


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



def time_plot(vod, site, year, out_path):
    """
    Creates the times series VOD plots  for a specif site and year.
    Makes 24h and 4h average for VOD_Raw and VOD (corrected).
    """
    print(f"Timeseries plot {site} and {year}")
    vod = vod[vod.index.year == int(year)]
    vod_ts_day = vod.groupby(pd.Grouper(freq='1D', level='Epoch')).mean()
    vod_ts_4h = vod.groupby(pd.Grouper(freq='4h', level='Epoch')).mean()

    # Figure of the time series with and without baseline
    vod_names = ['VOD_raw', 'VOD']

    fig = go.Figure()
    # Add two 4hour means
    i = 0
    col_4 = ["lightgrey", "lightgreen"]
    col = ["grey", "darkgreen"]
    for iname in vod_names:
        fig.add_trace(go.Scatter(
            x=vod_ts_4h.index,
            y=vod_ts_4h[iname],
            mode='lines+markers',
            name=iname + " 4h",
            line=dict(color=col_4[i])
        ))
        i = i + 1
    # Add two 1day means
    i = 0
    for iname in vod_names:
        fig.add_trace(go.Scatter(
            x=vod_ts_day.index,
            y=vod_ts_day[iname],
            mode='lines+markers',
            name=iname + " 1d",
            line=dict(color=col[i])
        ))
        i = i + 1

    # Update layout
    fig.update_layout(
        title=f'GNSS VOD at {site} Tower, {year}',
        xaxis_title='Date',
        yaxis_title='GNSS-VOD (L1)',
        xaxis=dict(
            tickformat='%d-%b'
        ),
        legend_title='Measurements',
        template='plotly_white'
    )

    # Save plot as an interactive HTML file
    file_name = f'{site}_vod_timeseries_{year}_avg.html'
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
    
    # Load and filter input files
    files_product = index_data_files(product_path, station_name, "all")
    files_product = filter_index_files(files_product, baseline=baseline, notincluded="old")
    ds_product = xr.open_dataset(files_product["File"][0]).to_dataframe()
    
    # Run times series plot
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

    # Run for each month of the year the hemispherical plot
    for month in range(1, 13):
        monthly_hemi_plot(df, station_name, year, month, out_path, is_baseline=True)

    print("Plots finished")
