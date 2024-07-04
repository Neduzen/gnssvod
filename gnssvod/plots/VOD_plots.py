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
from gnssvod.plots.VOD_timeseries_helper_functions import index_data_files


def monthly_hemi_plot(vod, patches, site, year, month, out_path):
    print(f"Hemisphere plot {site} and {year}-{month}")
    vod_avg = vod[vod.index.get_level_values('Epoch').month == month]
    vod_avg = vod_avg[vod_avg.index.get_level_values('Epoch').year == int(year)]
    vod_avg = vod_avg.groupby(['CellID']).mean(['mean', 'std', 'count'])

    # Make the VOD figure
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))

    # Associate the mean values to the patches, join inner will drop patches with no data, making plotting slightly faster
    ivod_data = vod_avg[f"VOD_mean"].where(vod_avg[f"VOD_count"] > 20)  # select minimal number of observations
    ipatches = pd.concat([patches, ivod_data], join='inner', axis=1)
    # Plotting with colored patches
    pc = PatchCollection(ipatches.Patches, array=ipatches[f"VOD_mean"], edgecolor='face', linewidth=1)
    pc.set_clim([-0.1, 3])
    ax.add_collection(pc)
    ax.set_rlim([0, 90])
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title(f'VOD - {year} - {month}', fontsize=20)

    plt.colorbar(pc, ax=ax, location='bottom', shrink=.5, pad=0.05, label='GNSS-VOD')
    plt.show()
    file_name = f'{site}_vod_hemplot_{year}-{month:02d}_avg.png'
    plot_path = os.path.join(out_path, file_name)
    fig.savefig(plot_path, bbox_inches='tight', pad_inches=0.1, dpi=300)

    # Plot using Plotly
    # fig = go.Figure()
    #
    # fig.add_trace(go.Scatterpolar(
    #     r=ipatches['r'],
    #     theta=ipatches['theta_degrees'],
    #     mode='markers',
    #     marker=dict(
    #         size=8,
    #         color=ipatches['VOD_mean'],
    #         colorscale='Viridis',
    #         cmin=-0.1,
    #         cmax=3,
    #         colorbar=dict(
    #             title='GNSS-VOD',
    #             orientation='h',
    #             x=0.5,
    #             xanchor='center',
    #             y=-0.2,
    #             len=0.5
    #         )
    #     )
    # ))
    #
    # fig.update_layout(
    #     polar=dict(
    #         radialaxis=dict(range=[0, 90]),
    #         angularaxis=dict(direction='clockwise')
    #     ),
    #     title=f'Laegern average VOD - {month}',
    #     showlegend=False
    # )
    #
    # # Save plot as an interactive HTML file
    # file_name = f'{site}_vod_hemplot_{month}_avg.html'
    # plot_path = os.path.join(out_path, file_name)
    # fig.write_html(plot_path)
    #
    # # Display plot in the notebook (if running in a Jupyter environment)
    # fig.show()


def time_plot(vod, site, year, out_path):
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
    #
    # # Save anomaly file as pickel
    # dump_path = os.path.join(out_path, f'{site}_anom_file.pkl')
    # with open(dump_path, 'wb') as file:
    #     pkl.dump(vod_anom, file)


def do_plot(timeseries_path, product_path, year, out_path):
    station_name = list(timeseries_path.keys())[0]
    timeseries_path = timeseries_path[station_name]
    product_path = product_path[station_name]
    out_path = out_path[station_name]

    files_product = index_data_files(product_path, station_name, "all")
    ds_product = xr.open_dataset(files_product[0]).to_dataframe()
    time_plot(ds_product, station_name, year, out_path)

    files = index_data_files(timeseries_path, station_name, "annual")

    files_names = files["File"][files["Year"] == year]

    if len(files_names) == 0:
        print(f"Error: No timeseries files found for site {station_name} and year {year}")
    if len(files_names) > 1:
        print(f"Error: More than 1 timeseries files found for site {station_name} and year {year}")
        print(f"Files: {files_names}")
    # Load the processed VOD data set
    ds = xr.open_dataset(files_names.iloc[0])
    # Convert the xarray to a pandas data frame, sorted by Epoch and satellite (Coordinates of ds). All Data variables of
    # ds_new are now columns in the data frame.
    df = ds.to_dataframe().dropna(how='all').reorder_levels(['Epoch', 'SV']).sort_index()
    # df = df[['VOD', 'Azimuth', 'Elevation']]


    # # Initialize hemispheric grid and patches
    hemi = gv.hemibuild(2)
    patches = hemi.patches()

    for month in range(1,13):
        monthly_hemi_plot(df, patches, station_name, year, month, out_path)

    # # # Classify vod into grid cells
    # vod = hemi.add_CellID(df, aziname='Azimuth', elename='Elevation').drop(columns=['Azimuth', 'Elevation'])
    # vod_avg = vod.groupby(['CellID']).agg(['mean', 'std', 'count'])
    # vod_avg.columns = ["_".join(x) for x in vod_avg.columns.to_flat_index()]
    # # ----------------------------------------------------------------------------------------------
    # monthly_hemi_plot(vod_avg, patches, station_name, month, out_path)

    # # Initialize hemispheric grid
    # hemi = gv.hemibuild(2)
    # # Get patches for plotting later
    # patches = hemi.patches()
    # # Classify vod into grid cells and drop the azm und ele columns after
    # vod = hemi.add_CellID(df, aziname='Azimuth', elename='Elevation').drop(columns=['Azimuth', 'Elevation'])
    # # Get mean, std and count values per grid cell
    # vod_avg = vod.groupby(['CellID']).agg(['mean', 'std', 'count'])
    # # flatten the columns
    # vod_avg.columns = ["_".join(x) for x in vod_avg.columns.to_flat_index()]


    # ----------------------------------------------------------------------------------------------
    # VOD time series whole year

    # fig, ax = plt.subplots(1, figsize=(10, 5))
    # for i, iname in enumerate(vod_names):
    #     # Plot each measurement and color by signal-to-noise ratio
    #     hs = ax.plot(vod_ts.index.get_level_values('Epoch'), vod_ts[iname], label=iname)
    #
    # myFmt = mdates.DateFormatter('%d-%b')
    # ax.xaxis.set_major_formatter(myFmt)
    # ax.set_ylabel('GNSS-VOD (L1)')
    # ax.legend()
    # plt.title(f'GNSS VOD at {site} Tower, {month}', fontsize=20)
    # plt.show()
    # file_name = f'{site}_vod_timeserie_{month}_avg.png'
    # plot_path = os.path.join(out_path, file_name)
    # fig.savefig(plot_path, bbox_inches='tight', pad_inches=0.1, dpi=300)

    print("Plots finished")
