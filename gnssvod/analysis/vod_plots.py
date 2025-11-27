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
from gnssvod.analysis.vod_dyn_plot import plot_dyn_hemispherical, features_plot, plot_eth_var_comparison



def monthly_hemi_plot(vod, site, year, month, out_path, is_baseline=False, do_anomaly=False, ele_lim=90):
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

    title = f'{vod_name} - {year} - {month}'
    plot_path = os.path.join(out_path, file_name)

    plot_hemisphere(vod_avg, vod_name, patches, z_lim, plot_path, title=title, ele_lim=ele_lim)


def plot_hemisphere(vod_avg, vod_name, patches, z_lim, plot_path, title="hem", ele_lim=90, colbar=None):
    # Make the VOD figure
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))

    # Associate the mean values to the patches, join inner will drop patches with no data, making plotting slightly faster
    ivod_data = vod_avg[vod_name].where(vod_avg[f"bl_VOD_count"] > 5)  # select minimal number of observations
    ipatches = pd.concat([patches, ivod_data], join='inner', axis=1)
    # Plotting with colored patches
    pc = PatchCollection(ipatches.Patches, array=ipatches[vod_name], edgecolor='face', linewidth=1)
    pc.set_clim(z_lim)
    ax.add_collection(pc)
    ax.set_rlim([0, ele_lim])
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title(title, fontsize=20)

    plt.colorbar(pc, ax=ax, location='bottom', shrink=.5, pad=0.05, label='GNSS-VOD')
    # plt.show()
    fig.savefig(plot_path, bbox_inches='tight', pad_inches=0.1, dpi=300)


def hemi_dif_plot(vod, site, year, month1, month2, out_path, is_baseline=False, do_anomaly=False):
    """
    Creates the monthly VOD hemispherical plot for a specific month and year and site.
    """
    print(f"Hemisphere plot {site} and {year}: {month1}-{month2}")
    vod_name = "VOD"
    file_name = f'{site}_vod_hemplot_dif_{year}_{month1:02d}-{month2:02d}_avg.png'
    z_lim = [-0.4,0.1]

    # # Initialize hemispheric grid and patches
    hemi = gv.hemibuild(2)
    patches = hemi.patches()

    if is_baseline:
        vod_avg1 = (vod[vod.index.get_level_values('Date').month == month1].
                   rename(columns={"VOD_mean": "VOD", "VOD_count": "bl_VOD_count"}))
        vod_avg2 = (vod[vod.index.get_level_values('Date').month == month2].
                   rename(columns={"VOD_mean": "VOD", "VOD_count": "bl_VOD_count"}))
    else:
        # vod_avg1 = vod[vod.index.get_level_values('Epoch').month == month1]
        # vod_avg1 = vod_avg1[vod_avg1.index.get_level_values('Epoch').year == int(year)]
        # vod_avg2 = vod[vod.index.get_level_values('Epoch').month == month2]
        # vod_avg2 = vod_avg2[vod_avg2.index.get_level_values('Epoch').year == int(year)]
        vod_avg1 = vod[vod.index.get_level_values('Epoch').month <= 2]
        vod_avg2 = vod[vod.index.get_level_values('Epoch').month > 5]
        vod_avg2 = vod_avg2[vod_avg2.index.get_level_values('Epoch').month < 8]

    vod_avg1 = vod_avg1.groupby(['CellID']).mean(['mean', 'std', 'count'])
    vod_avg2 = vod_avg2.groupby(['CellID']).mean(['mean', 'std', 'count'])

    vod_avg = vod_avg1.copy()
    vod_avg["VOD"][:] = vod_avg2["VOD"][:] - vod_avg1["VOD"][:]
    vod_avg["VOD_raw"][:] = vod_avg2["VOD_raw"][:] - vod_avg1["VOD_raw"][:]

    title = f'{vod_name} {year} : {month1} - {month2} '
    plot_path = os.path.join(out_path, file_name)
    plot_hemisphere(vod_avg1, "VOD_raw", patches, [-0.1, 3.0], plot_path, title=title, ele_lim=60)
    plot_hemisphere(vod_avg2, "VOD_raw", patches, [-0.1, 3.0], plot_path, title=title, ele_lim=60)

    plot_hemisphere(vod_avg, "VOD_raw", patches, z_lim, plot_path, title=title, ele_lim=60)


def hemi_dif_plot2(vod1, vod2, site, name1, name2, out_path, do_anomaly=False):
    """
    Creates the monthly VOD hemispherical plot for a specific month and year and site.
    """
    print(f"Hemisphere plot {site}: {name1}-{name2}")
    vod_name = "VOD"
    file_name = f'{site}_vod_hemplot_dif_{name1}-{name2}_avg.png'
    z_lim = [-0.5, 0.5]

    # # Initialize hemispheric grid and patches
    hemi = gv.hemibuild(2)
    patches = hemi.patches()

    vod_avg1 = vod1.groupby(['CellID']).mean(['mean', 'std', 'count'])
    vod_avg2 = vod2.groupby(['CellID']).mean(['mean', 'std', 'count'])

    # # Find the intersection of the indices
    common_indices = vod_avg1.index.intersection(vod_avg2.index)

    # Filter both DataFrames to keep only the common indices
    vod_avg1_filtered = vod_avg1.loc[common_indices]
    vod_avg2_filtered = vod_avg2.loc[common_indices]

    vod_avg = vod_avg1_filtered.copy()
    vod_avg["VOD"][:] = vod_avg1_filtered["VOD"][:] - vod_avg2_filtered["VOD"][:]
    vod_avg["VOD_raw"][:] = vod_avg1_filtered["VOD_raw"][:] - vod_avg2_filtered["VOD_raw"][:]

    title = f'{vod_name}: {name1} - {name2} '
    file_name = f'{site}_vod_hemplot_dif_JulAug21.png'
    plot_path = os.path.join(out_path, file_name)
    plot_hemisphere(vod_avg1, "VOD_raw", patches, [-0.1, 3], plot_path, title=f"VOD: July and August 2021", ele_lim=60)
    file_name = f'{site}_vod_hemplot_dif_JulAug22.png'
    plot_path = os.path.join(out_path, file_name)
    plot_hemisphere(vod_avg2_filtered, "VOD_raw", patches, [-0.1, 3], plot_path, title=f"VOD: July and August 2022",
                    ele_lim=60)

    title = f'{vod_name}: {name1} - {name2} '
    plot_path = os.path.join(out_path, file_name)
    plot_hemisphere(vod_avg1, "VOD_raw", patches, [-0.1, 3.0], plot_path, title=title, ele_lim=60)
    plot_hemisphere(vod_avg2, "VOD_raw", patches, [-0.1, 3.0], plot_path, title=title, ele_lim=60)

    plot_hemisphere(vod_avg, "VOD_raw", patches, z_lim, plot_path, title=title, ele_lim=60)



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
    vod_names = ['VOD_raw', 'VOD']

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



def do_plot(timeseries_path, product_path, baseline_path, year, baseline, out_path, max_elevation=90, ignore_glosas=False):
    """
    Creates all needed plots for a site for a year.
    """
    station_name = list(timeseries_path.keys())[0]
    timeseries_path = timeseries_path[station_name]
    product_path = product_path[station_name]
    baseline_path = baseline_path[station_name]
    out_path = out_path[station_name]

    print(f"Plot gnss site {station_name} at {out_path}")

    houragg = "1h"
    filename_ext = ""
    exclude = ["_const"]
    exclude_bl = "_noR"
    if ignore_glosas:
        filename_ext = f"_noR_{houragg}"
        exclude.append("")
        exclude_bl = ""
    else:
        filename_ext = f"_{houragg}"
        exclude.append("_noR")
    if max_elevation < 90:
        filename_ext = f"{filename_ext}_maxEle{max_elevation}"
        exclude.append("")
    else:
        exclude.append(f"_maxEle")
    # file_dates = filter_index_files(file_dates, baseline=baseline, text=filename_ext, notincluded="_noR")

    print(f"Filename: {filename_ext}, exclude: {exclude} at {product_path}")
    # Load and filter input files
    files_product = index_data_files(product_path, station_name, "all")
    files_product = filter_index_files(files_product, baseline=baseline, text=filename_ext, notincluded=exclude[0])
    files_product = filter_index_files(files_product, baseline=baseline, text=filename_ext, notincluded=exclude[1])
    files_product = filter_index_files(files_product, baseline=baseline, text=filename_ext, notincluded=exclude[2])

    print(f"files: {files_product['File'][0]}")
    ds_product = xr.open_dataset(files_product["File"][0]).to_dataframe()

    # Run times series plot
    time_plot(ds_product, station_name, None, out_path)
    time_plot(ds_product, station_name, year, out_path)

    # Monthly VOD plot
    files_bl = index_data_files(baseline_path, station_name, "annual")
    files_names_bl = filter_index_files(files_bl, year=year, baseline=baseline, notincluded=exclude_bl)["File"]

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
        monthly_hemi_plot(df, station_name, year, month, out_path_hem, is_baseline=True, ele_lim=max_elevation)

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


def site_comparison_plot(product_path, baseline, out_path, ignore_glosas=False, max_elevation=90):
    import xarray as xr
    import pandas as pd
    import plotly.graph_objects as go
    import os

    vod_name = "VOD"  # Name of your VOD variable in the dataset
    # out_path should be defined, e.g., out_path = "/path/to/output/plots"

    filename_ext = ""
    exclude = "_noR"
    if ignore_glosas:
        filename_ext = "_noR"
        exclude = ""
    if max_elevation < 90:
        filename_ext = f"_maxEle{max_elevation}{filename_ext}"
    else:
        exclude = f"_maxEle{exclude}"

    files_product_1 = filter_index_files(index_data_files(product_path, "Dav", "all"), baseline=baseline,
                                         text=filename_ext, notincluded=exclude)
    vod_ds1 = xr.open_dataset(files_product_1["File"][0])

    # Convert to DataFrame and process
    vod_df1 = vod_ds1.to_dataframe()
    vod_high_freq1 = vod_df1.groupby(pd.Grouper(freq='1D', level='Epoch')).mean()
    vod_low_freq1 = vod_high_freq1.rolling('14D', center=True).mean()
    vod_low_freq1["doy"] = vod_low_freq1.index.dayofyear
    vod_low_freq1["year"] = vod_low_freq1.index.year

    vod_mean1 = vod_high_freq1.rolling('30D', center=True).mean()
    vod_mean1["doy"] = vod_mean1.index.dayofyear
    vod_mean1["year"] = vod_mean1.index.year
    vod_mean1["VOD_mean"] = vod_mean1[vod_name]  # Use actual vod_name
    vod_mean1 = vod_mean1.groupby(vod_mean1["doy"]).mean().reset_index()
    vod_mean1["doy"] = vod_mean1["doy"].astype(float)  # Ensure DOY is numeric for merging

    # --- Load and Prepare Dataset 2 (Blue Colors) ---
    another_product_path = r"S:\group\rsws_gnss\VOD_product_live\Laeg"
    files_product_2 = filter_index_files(index_data_files(another_product_path, "Laeg", "all"), baseline=baseline,
                                         text=filename_ext, notincluded=exclude)
    vod_ds2 = xr.open_dataset(files_product_2["File"][0])

    # Convert to DataFrame and process
    vod_df2 = vod_ds2.to_dataframe()
    vod_high_freq2 = vod_df2.groupby(pd.Grouper(freq='1D', level='Epoch')).mean()
    vod_low_freq2 = vod_high_freq2.rolling('14D', center=True).mean()
    vod_low_freq2["doy"] = vod_low_freq2.index.dayofyear
    vod_low_freq2["year"] = vod_low_freq2.index.year

    vod_mean2 = vod_high_freq2.rolling('30D', center=True).mean()
    vod_mean2["doy"] = vod_mean2.index.dayofyear
    vod_mean2["year"] = vod_mean2.index.year
    vod_mean2["VOD_mean"] = vod_mean2[vod_name]  # Use actual vod_name
    vod_mean2 = vod_mean2.groupby(vod_mean2["doy"]).mean().reset_index()
    vod_mean2["doy"] = vod_mean2["doy"].astype(float)  # Ensure DOY is numeric for merging

    # --- Plotting ---
    print(f"Full timeseries plot of comparing two datasets")
    title = f'GNSS VOD Timeseries at  (Dataset 1 vs Dataset 2)'
    tick_format = '%y-%b'
    file_name = f'vod_timeseries_dual_dataset.html'  # New filename for dual plot

    fig = go.Figure()

    # Define color scales for each dataset
    red_colors = ['rgb(255, 10, 10)', 'rgb(220, 10, 10)', 'rgb(180, 10, 10)', 'rgb(150, 10, 10)',
                  'rgb(100, 10, 10)']  # Example shades of red
    blue_colors = ['rgb(10, 10, 255)', 'rgb(10, 10, 220)', 'rgb(10, 10, 180)', 'rgb(10, 10, 150)',
                   'rgb(10, 10, 100)']  # Example shades of blue

    # Add traces for Dataset 1 (red colors)
    years_to_plot = sorted(list(set(vod_low_freq1["year"].dropna())))  # Get years from the data
    if len(years_to_plot) > len(red_colors):
        print("Warning: More years than predefined red colors. Colors will cycle.")
        # Or, generate more colors dynamically: px.colors.sequential.Reds
    for i, yeari in enumerate(years_to_plot):
        vodplot = vod_low_freq1[vod_low_freq1.year == yeari]
        fig.add_trace(go.Scatter(
            x=vodplot.doy,
            y=vodplot[vod_name],
            mode='lines',
            name=f'Dav {yeari}',
            line=dict(color=red_colors[i % len(red_colors)])  # Cycle through colors
        ))

    # Add mean for Dataset 1 (black)
    fig.add_trace(go.Scatter(
        x=vod_mean1["doy"],
        y=vod_mean1[vod_name],
        mode='lines',
        name=f'Dav Mean',
        line=dict(color="#4D2829", width=3)

    ))

    # Add traces for Dataset 2 (blue colors)
    years_to_plot = sorted(list(set(vod_low_freq2["year"].dropna())))  # Get years from the data
    if len(years_to_plot) > len(blue_colors):
        print("Warning: More years than predefined blue colors. Colors will cycle.")
        # Or, generate more colors dynamically: px.colors.sequential.Blues
    for i, yeari in enumerate(years_to_plot):
        vodplot = vod_low_freq2[vod_low_freq2.year == yeari]
        fig.add_trace(go.Scatter(
            x=vodplot.doy,
            y=vodplot[vod_name],
            mode='lines',
            name=f'Laeg {yeari}',
            line=dict(color=blue_colors[i % len(blue_colors)])  # Cycle through colors
        ))

    # Add mean for Dataset 2 (gray)
    fig.add_trace(go.Scatter(
        x=vod_mean2["doy"],
        y=vod_mean2[vod_name],
        mode='lines',
        name=f'Laeg Mean',
        line=dict(color="#425B80", width=3)
    ))

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Day of Year (DOY)',  # Changed to DOY as you're plotting by DOY
        yaxis_title=f'GNSS-VOD (L1) ({vod_name})',
        legend_title='Measurements',
        template='plotly_white',
        hovermode="x unified"  # Improve hover experience
    )

    # Save the plot
    plot_path = os.path.join(out_path, file_name)
    fig.write_html(plot_path)

    fig.show()  # Display the plot


def analysis_plot(product_path, baseline, out_path, ignore_glosas=False, max_elevation=90):
    site = list(product_path.keys())[0]
    product_path = product_path[site]
    out_path = out_path[site]


    # ETH new data test
    hempathy1 = r"S:\group\rsws_gnss\VOD_hemisphere_live\Laeg\Laeg_2025_VOD_hemisphere_bl15days_noR_30min.nc"
    vod25 = xr.open_dataset(hempathy1).to_dataframe()
    plot_eth_var_comparison(vod25)

    # HEM DIF LAEG 2021 - 2022 summer
    hempathy1 = r"S:\group\rsws_gnss\VOD_hemisphere_live\Laeg\Laeg_2021_VOD_hemisphere_bl15days_noR_30min.nc"
    hempathy2 = r"S:\group\rsws_gnss\VOD_hemisphere_live\Laeg\Laeg_2022_VOD_hemisphere_bl15days_noR_30min.nc"
    out_path_hem = os.path.join(out_path, "hemplotdif")
    name1 = "July & August 2021"
    name2 = "July & August 2022"

    vod1 = xr.open_dataset(hempathy1).to_dataframe()
    features_plot(vod1, "VOD_raw")
    vod1 = vod1[vod1.index.get_level_values('Epoch').dayofyear >= 190]
    vod1 = vod1[vod1.index.get_level_values('Epoch').dayofyear <= 230]
    plot_dyn_hemispherical(vod1, out_path_hem, 'VOD_raw')

    vod2 = xr.open_dataset(hempathy2).to_dataframe()
    vod2 = vod2[vod2.index.get_level_values('Epoch').dayofyear >= 190]
    vod2 = vod2[vod2.index.get_level_values('Epoch').dayofyear <= 230]


    hemi_dif_plot2(vod1, vod2, site, name1, name2, out_path_hem)

    # Load and filter input files
    # files_product = index_data_files(product_path, site, "all")
    # files_product = filter_index_files(files_product, baseline=baseline, text=filename_ext, notincluded=exclude)
    hempathy = r"S:\group\rsws_gnss\VOD_hemisphere_live\Laeg\Laeg_2024_VOD_hemisphere_bl15days_noR_30min.nc"
    vod = xr.open_dataset(hempathy).to_dataframe()
    out_path_hem = os.path.join(out_path, "hemplotdif")
    hemi_dif_plot(vod, site, 2024, 12, 6, out_path_hem, is_baseline=False)
    # monthly_hemi_plot(df, station_name, year, month, out_path_hem, is_baseline=True)


    site_comparison_plot(product_path, baseline, out_path, ignore_glosas=False, max_elevation=90)

    import pickle

    # p1 = r"C:\Users\michni\Downloads\Dav_VOD_atS2overpass.pkl"
    # pon = r"S:\group\rsws_gnss\VOD_product_live\Dav\Dav_VOD_product_bl15_1h.nc"
    # pn = r"S:\group\rsws_gnss\VOD_product_live\Dav\Dav_VOD_product_bl15_1h_maxEle60.nc"
    # pn2 = r"S:\group\rsws_gnss\VOD_product_live\Dav\Dav_VOD_product_bl15_1h_maxEle60 - Copy.nc"
    # pn3 = r"S:\group\rsws_gnss\VOD_product_live\Dav\Dav_VOD_product_bl15_1h_maxEle60_noR.nc"
    #
    # pn3 = xr.open_dataset(pn3).to_dataframe()
    # pn3 = pn3[pn3.index.hour == 11]
    # pn2 = xr.open_dataset(pn2).to_dataframe()
    # pn2 = pn2[pn2.index.hour == 11]
    # pn = xr.open_dataset(pn).to_dataframe()
    # pn = pn[pn.index.hour == 11]
    # pon = xr.open_dataset(pon).to_dataframe()
    # pon = pon[pon.index.hour == 11]
    # po = pd.read_pickle(p1)
    #
    # name = ["new", "old_new procc", "jasi"]
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(
    #     x=po.index,
    #     y=po["VOD_anom_corr"],
    #     mode='lines',
    #     name=f'jasi'
    # ))
    # fig.add_trace(go.Scatter(
    #     x=pon.index,
    #     y=pon["VOD"],
    #     mode='lines',
    #     name=f'old_new procc'
    # ))
    # fig.add_trace(go.Scatter(
    #     x=pn.index,
    #     y=pn["VOD"],
    #     mode='lines',
    #     name=f'new'
    # ))
    # fig.add_trace(go.Scatter(
    #     x=pn.index,
    #     y=pn["VOD_anom_corr"],
    #     mode='lines',
    #     name=f'new VOD_anom_corr'
    # ))
    # fig.add_trace(go.Scatter(
    #     x=pn2.index,
    #     y=pn2["VOD_anom_corr"],
    #     mode='lines',
    #     name=f'new VOD_anom_corr 2'
    # ))
    # fig.add_trace(go.Scatter(
    #     x=pn2.index,
    #     y=pn2["VOD"],
    #     mode='lines',
    #     name=f'new 2'
    # ))
    # fig.add_trace(go.Scatter(
    #     x=pn3.index,
    #     y=pn3["VOD_anom_corr"],
    #     mode='lines',
    #     name=f'new norus corr'
    # ))
    # fig.add_trace(go.Scatter(
    #     x=pn3.index,
    #     y=pn3["VOD_raw"],
    #     mode='lines',
    #     name=f'new norus'
    # ))
    # fig.show()
    # # # Update layout
    # # fig.update_layout(
    # #     title=title,
    # #     xaxis_title='Date',
    # #     yaxis_title='GNSS-VOD (L1)',
    # #     xaxis=dict(
    # #         tickformat=tick_format
    # #     ),
    # #     legend_title='Measurements',
    # #     template='plotly_white'
    # # )


    filename_ext = ""
    exclude = "_noR"
    if ignore_glosas:
        filename_ext = "_noR"
        exclude = ""
    if max_elevation < 90:
        filename_ext = f"_maxEle{max_elevation}{filename_ext}"
    else:
        exclude = f"_maxEle{exclude}"

    # Load and filter input files
    files_product = index_data_files(product_path, site, "all")
    files_product = filter_index_files(files_product, baseline=baseline, text=filename_ext, notincluded=exclude)
    vod = xr.open_dataset(files_product["File"][0]).to_dataframe()


    print(f"Full timeseries plot of {site}")
    title = f'GNSS VOD timeseries at {site}'
    iname="VOD"
    file_name = f'{site}_vod_timeseries_annual_avg.html'
    file_name_lfq = f'{site}_vod_timeseries_annual_avg_lowfreq.html'
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

    vod_mean_low_freq2 = vod_low_freq2.rolling('14D', center=True).mean()
    vod_mean_low_freq2["doy"]=vod_mean_low_freq2.index.dayofyear
    vod_mean_low_freq2["year"]=vod_mean_low_freq2.index.year
    vod_mean_low_freq2["VOD_mean"]=vod_mean_low_freq2["VOD"]
    vod_mean_low_freq2=vod_mean_low_freq2.groupby(vod_mean_low_freq2["doy"]).mean().reset_index()
    vod_mean_low_freq2["doy"]=vod_mean_low_freq2.index

    vod_low_freq2 = pd.merge(vod_low_freq2, vod_mean[["VOD_mean", "doy"]], how="left", on=["doy", "doy"])




    fig = go.Figure()
    # Add two 1day means
    i = 0
    for yeari in [2021,2022,2023,2024,2025]:
        vodplot=vod_low_freq2[vod_low_freq2.year==yeari]
        fig.add_trace(go.Scatter(
            x=vodplot.doy,
            y=vodplot[iname],
            mode='lines',
            name=f'{yeari}'
        ))
        i = i + 1
    fig.add_trace(go.Scatter(
        x=vod_mean_low_freq2.index,
        y=vod_mean_low_freq2[iname],
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
    plot_path = os.path.join(out_path, file_name_lfq)
    fig.write_html(plot_path)

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


def compare_products_at_site(files_product, site, out_path, year):
    fig = go.Figure()
    for pathi in files_product["File"]:
        mname = pathi.split(r'bl15_')[1].split('.nc')[0]
        vod = xr.open_dataset(pathi).to_dataframe()

        if out_path is None:
            raise ValueError("No out_path defined")
        if year is None:
            print(f"Full timeseries plot of {site}")
            title = f'GNSS VOD timeseries at {site}'
            file_name = f'{site}_vod_timeseries_comp.html'
            high_freq = '1D'
            low_freq = '7D'
            tick_format = '%y-%b'
            vod_high_freq = vod.groupby(pd.Grouper(freq=high_freq, level='Epoch')).mean()
            vod_low_freq = vod_high_freq.rolling('7D', center=True).mean()
        else:
            print(f"Timeseries plot {site} and {year}")
            high_freq = '4h'
            low_freq = '1D'
            tick_format = '%d-%b'
            title = f'GNSS VOD at {site} of {year}'
            file_name = f'{site}_vod_timeseries_{year}_avg_compare.html'
            vod = vod[vod.index.year == int(year)]
            vod_high_freq = vod.groupby(pd.Grouper(freq=high_freq, level='Epoch')).mean()
            vod_low_freq = vod_high_freq.rolling('7D', center=True).mean()

        # Figure of the time series with and without baseline
        vod_names = ['VOD_raw', 'VOD']

        # Add two 4hour means
        i = 0
        col_4 = ["lightgrey", "lightgreen"]
        col = ["grey", "darkgreen"]
        for iname in vod_names:
            fig.add_trace(go.Scatter(
                x=vod_high_freq.index,
                y=vod_high_freq[iname],
                mode='lines+markers',
                name=f'{iname} {mname} {high_freq}'
            ))
            i = i + 1
        # Add two 1day means
        i = 0
        for iname in vod_names:
            fig.add_trace(go.Scatter(
                x=vod_low_freq.index,
                y=vod_low_freq[iname],
                mode='lines',
                name=f'{iname} {mname} {low_freq}'
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


def plot_points_hemisphere(vod_avg, vod_name, z_lim, plot_path, title="hem", ele_lim=90, colbar=None):
    # Make the VOD figure
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
    custom_colorscale = [
        [0.0, 'rgb(0, 0, 0)'],  # Black
        [0.5, 'rgb(0, 100, 0)'],  # Dark Green
        [1.0, 'rgb(144, 238, 144)']  # Light Green
    ]
    # Associate the mean values to the patches, join inner will drop patches with no data, making plotting slightly faster
    vod_avg = vod_avg[vod_avg[vod_name]>-3]  # select minimal number of observations
    # --- 2. Create the Figure ---
    fig = go.Figure(go.Scatterpolar(
        r=vod_avg["Elevation"],
        theta=vod_avg["Azimuth"],
        mode='markers',
        marker=dict(
            color=vod_avg[vod_name] ,  # This is your 'value'
            showscale=True,  # Show the color bar
            size=7,
            colorbar_title=vod_name,
        colorscale = custom_colorscale
        )
    ))

    # --- 3. Customize the Layout ---
    # This makes it look like a standard compass/hemisphere plot
    fig.update_layout(
        title=title,
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 90]  # Set the elevation range (0 to 90 degrees)
            ),
            angularaxis=dict(
                # Set 0 degrees to be North
                rotation=90,
                # Set direction to clockwise
                direction="clockwise",
                # Show compass directions
                tickvals=[0, 90, 180, 270],
                ticktext=["N", "E", "S", "W"]
            )
        )
    )

    fig.write_html(plot_path)


def plot_hemi_sat_constellation(station_name, days=15):
    vod = xr.open_mfdataset(r"S:\group\rsws_gnss\VOD_t2\Lae2\vod_Lae2_20250801_20250831.nc")
    import gnssvod.hemistats.hemistats as hemistats
    #
    vod_week_sel = vod.to_dataframe()
    vod_week_sel = vod_week_sel[vod_week_sel.index.get_level_values('Epoch').day < days]
    # Initialize hemispheric grid
    hemi = hemistats.hemibuild(2)
    # # Classify vod data into grid cells and drop the azm und ele columns after
    vod_week = hemi.add_CellID(vod_week_sel, aziname='Azimuth', elename='Elevation')
    first_letters = [sv[0] for sv in vod_week.index.get_level_values('SV').values]
    vod_week['SV_type'] = first_letters
    svtypes = pd.unique(first_letters)

    out_path_hem = r"C:\Users\michni\Documents\temp_plot"
    # Run for each month of the year the hemispherical plot
    for sv in svtypes:
        vod_week_sv = vod_week[vod_week["SV_type"] == sv]
        vod_week_sv = vod_week_sv.reset_index('SV')
        n = len(vod_week_sv)
        print(f"{sv}: {n}")
        fname = fr'{out_path_hem}\{station_name}_vod_{sv}.html'

        plot_points_hemisphere(vod_week_sv, "VOD", [0, 3], fname, title=f"{sv}: {n}", ele_lim=80)


def calcALL(low_freq='4D', vod_names = ['VOD_raw'], t=''):

    lae80noR = r'S:\\group\\rsws_gnss\\VOD_product_live\\Laeg\\Laeg_VOD_product_bl15_1h_maxEle80_noR.nc'
    lae1 = r'S:\\group\\rsws_gnss\\VOD_product_live\\Laeg\\Laeg_VOD_product_bl15_1h_maxEle80.nc'
    lae1h = r'S:\\group\\rsws_gnss\\VOD_product_live\\Laeg\\Laeg_VOD_product_bl15_1h.nc'
    dav80noR = r'S:\\group\\rsws_gnss\\VOD_product_live\\Dav\\Dav_VOD_product_bl15_1h_maxEle80_noR.nc'
    dav1 = r'S:\\group\\rsws_gnss\\VOD_product_live\\Dav\\Dav_VOD_product_bl15_1h_maxEle80.nc'
    dav1h = r'S:\\group\\rsws_gnss\\VOD_product_live\\Dav\\Dav_VOD_product_bl15_1h.nc'
    plot_path = fr"C:\Users\michni\Documents\temp_plot\alldata_{low_freq}_{t}.html"
    sites = [("Lae-e80-noR", lae80noR), ("Dav-e80-noR", dav80noR),("Lae-e80", lae1), ("Dav-e80", dav1), ("Lae", lae1h), ("Dav", dav1h)]

    fig = go.Figure()
    high_freq = '12H'
    tick_format = '%y-%b'
    # Figure of the time series with and without baseline


    for site in sites:
        sname = site[0]
        fpath = site[1]
        vod = xr.open_dataset(fpath).to_dataframe()

        vod_high_freq = vod.groupby(pd.Grouper(freq=high_freq, level='Epoch')).mean()
        vod_low_freq = vod_high_freq.rolling(low_freq, center=True).mean()

        #
        # # Add two 4hour means
        # i = 0
        # col_4 = ["lightgrey", "lightgreen"]
        # col = ["grey", "darkgreen"]
        # for iname in vod_names:
        #     fig.add_trace(go.Scatter(
        #         x=vod_high_freq.index,
        #         y=vod_high_freq[iname],
        #         mode='lines+markers',
        #         name=f'{sname} {iname} {high_freq}'
        #     ))
        #     i = i + 1
        # Add two 1day means
        i = 0
        for iname in vod_names:
            fig.add_trace(go.Scatter(
                x=vod_low_freq.index,
                y=vod_low_freq[iname],
                mode='lines',
                name=f'{sname} {iname} {low_freq}'
            ))
            i = i + 1

        # Update layout
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='GNSS-VOD (L1)',
            xaxis=dict(
                tickformat=tick_format
            ),
            legend_title='Measurements',
            template='plotly_white'
        )

        # Save plot as an interactive HTML file
        fig.write_html(plot_path)


def plot_constellations(station_name, product_path, baseline, low_freq='7D', vod_names = ['VOD_raw', 'VOD']):

    plot_path = fr"C:\Users\michni\Documents\temp_plot\{station_name}_constellations_{low_freq}.html"

    files_product = index_data_files(product_path, station_name, "all")
    files_product = filter_index_files(files_product, baseline=baseline, text="_const", notincluded="")


    fig = go.Figure()
    high_freq = '6H'
    tick_format = '%y-%b'

    for fpath in files_product['File']:
        constellation = fpath.split(r'const')[1][0]
        vod = xr.open_dataset(fpath).to_dataframe()

        count = vod["Count"].sum()
        vod_high_freq = vod.groupby(pd.Grouper(freq=high_freq, level='Epoch')).mean()
        vod_low_freq = vod_high_freq.rolling(low_freq, center=True).mean()

        for iname in vod_names:
            fig.add_trace(go.Scatter(
                x=vod_low_freq.index,
                y=vod_low_freq[iname],
                mode='lines',
                name=f'{constellation}: {station_name}-{iname}-{low_freq}, n={count}'
            ))

        # Update layout
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='GNSS-VOD (L1)',
            xaxis=dict(
                tickformat=tick_format
            ),
            legend_title='Measurements',
            template='plotly_white'
        )

    # Save plot as an interactive HTML file
    fig.write_html(plot_path)
    print(f"Sat Const Plot saved to {plot_path}")