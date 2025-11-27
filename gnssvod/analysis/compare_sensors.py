import os
import time
import glob
import datetime
import numpy as np
import pandas as pd
import xarray as xr
import warnings
import gnssvod as gv
import glob
import plotly.graph_objects as go
# import gnssvod.gnssvod.hemistats.hemistats as hemistats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from gnssvod.analysis.vod_dyn_plot import plot_dyn_hemispherical

from gnssvod.analysis.vod_timeseries_helper_functions import index_data_files, filter_index_files


def test_sat_const():
    path = r'C:\Users\michni\Documents\temp_plot'

    tpath = r'S:\group\rsws_gnss\Paired_t2\DavCompareTwr\DavCompareTwr_20250913000000_20250914000000.nc'
    data_twr = xr.open_mfdataset(tpath).to_dataframe().dropna(how='all')

    # Dav paired
    dav1 = r'S:\group\rsws_gnss\Paired_t2\Dav\Dav_20251001000000_20251002000000.nc'
    dav2 = r'S:\group\rsws_gnss\Paired_t2\Dav2\Dav2_20251001000000_20251002000000.nc'

    ref1 = "Dav2_Twr"
    grnd1 = "Dav1_Grnd"
    ref2 = "Dav3T"
    grnd2 = "Dav4G"

    data_old = xr.open_mfdataset(
        dav1).to_dataframe().dropna(
        how='all')
    data_new = xr.open_mfdataset(
        dav2).to_dataframe().dropna(
        how='all')

    iref = data_old.xs(ref1, level='Station')
    igrn = data_old.xs(grnd1, level='Station')
    idat_old = iref.merge(igrn, on=['Epoch', 'SV'], suffixes=['_ref', '_grn'])

    iref = data_new.xs(ref2, level='Station')
    igrn = data_new.xs(grnd2, level='Station')
    idat_new = iref.merge(igrn, on=['Epoch', 'SV'], suffixes=['_ref', '_grn'])

    idat_all = idat_old.merge(idat_new, on=['Epoch', 'SV'], suffixes=['_old', '_new'])

    galileo = idat_all.copy().reset_index('SV').dropna()
    first_letters = [sv[0] for sv in galileo['SV'].values]
    galileo['SV_type'] = first_letters
    galileo = galileo[galileo['SV_type'] == "E"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=galileo.index[:1000],
        y=galileo['S1_ref_new'][:1000],
        mode='markers',
        name=f'ref new',  # Name for the legend (this will be for the trace itself, not the colorbar)
    ))
    fig.add_trace(go.Scatter(
        x=galileo.index[:1000],
        y=galileo['S1_ref_old'][:1000],
        mode='markers',
        name=f'ref old',  # Name for the legend (this will be for the trace itself, not the colorbar)
    ))

    fig.add_trace(go.Scatter(
        x=galileo.index[:1000],
        y=galileo['S1_grn_new'][:1000],
        mode='markers',
        name=f'grn new',  # Name for the legend (this will be for the trace itself, not the colorbar)
    ))
    fig.add_trace(go.Scatter(
        x=galileo.index[:1000],
        y=galileo['S1_grn_old'][:1000],
        mode='markers',
        name=f'grn old',  # Name for the legend (this will be for the trace itself, not the colorbar)
    ))
    # Update layout
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='SNR',
        title='SNR at Dav',
        template='plotly_white'
    )
    fig.show()
    fig.write_html(fr'{path}\Dav_allSensor_SNR.html')
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=galileo['S1_grn_new'][:1000],
        y=galileo['S1_ref_new'][:1000],
        mode='markers',
        name=f'new',  # Name for the legend (this will be for the trace itself, not the colorbar)
    ))
    fig.add_trace(go.Scatter(
        x=galileo['S1_grn_old'][:1000],
        y=galileo['S1_ref_old'][:1000],
        mode='markers',
        name=f'old',  # Name for the legend (this will be for the trace itself, not the colorbar)
    ))

    # Update layout
    fig.update_layout(
        xaxis_title='SNR grnd',
        yaxis_title='SNR ref',
        title='SNR at Dav',
        template='plotly_white'
    )
    fig.show()
    fig.write_html(fr'{path}\Dav_allSensor_SNR_cor.html')


    # MULTIPATH

    ref = "Dav2_Twr"
    grnd = "Dav3T"

    iref = data_twr.xs(ref, level='Station')
    igrn = data_twr.xs(grnd, level='Station')
    idat = iref.merge(igrn, on=['Epoch', 'SV'], suffixes=['_ref', '_grn'])

    # LAE
    data_twr = (xr.open_mfdataset(
        r'S:\group\rsws_gnss\Paired_t2\LaeCompare\LaeCompare_20250712000000_20250713000000.nc').
        to_dataframe().dropna(how='all'))

    ref = "Lae3T"
    grnd = "Laeg2_Twr"

    iref = data_twr.xs(ref, level='Station')
    igrn = data_twr.xs(grnd, level='Station')
    idat = iref.merge(igrn, on=['Epoch', 'SV'], suffixes=['_ref', '_grn'])


    multipath(idat, "E")


def multipath(gnss, sv_galileo = "E"):
    path = r'C:\Users\michni\Documents\temp_plot'
    # MP1 = C1 - (L1 * wavelength1)
    galileo = gnss.copy().reset_index('SV')
    first_letters = [sv[0] for sv in galileo['SV'].values]
    galileo['SV_type'] = first_letters
    galileo = galileo[galileo['SV_type'] == sv_galileo]
    galileo["MP1_ref"] = galileo["C1_ref"] - (galileo["L1_ref"] * 0.19046)  # 1575.420)
    galileo["MP1_grn"] = galileo["C1_grn"] - (galileo["L1_grn"] * 0.19046)

    # Frequencies of L1, L2 (GPS L1: 1575.42 MHz, L2: 1227.60 MHz)
    f1 = 1575.42
    f2 = 1227.60
    galileo["MP1_ref_double"] = galileo["C1_ref"] - (
                ((f1 ** 2 + f2 ** 2) / (f1 ** 2 + f2 ** 2)) * galileo["L1_ref"]) + (
                                            ((2 * f2 ** 2) / ((f1 ** 2) - (f2 ** 2))) * galileo["L2_ref"])

    lambda_L1 = 299792458 / 1575.42e6  # Speed of light / L1 frequency
    galileo['L1_meters'] = galileo['L1_ref'] * lambda_L1
    galileo['MP1_ref2'] = galileo['C1_ref'] - galileo['L1_meters']
    galileo['L1grn_meters'] = galileo['L1_grn'] * lambda_L1
    galileo['MP1_grn2'] = galileo['C1_grn'] - galileo['L1grn_meters']

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=galileo['Elevation_ref'][:1000],
        y=galileo['MP1_ref2'][:1000],
        mode='markers',
        name=f'SV Type: {sv_galileo}',  # Name for the legend (this will be for the trace itself, not the colorbar)
        marker=dict(
            size=8,
            opacity=0.7,
            color=galileo['S1_ref'][:1000],  # Data that determines the color
            colorscale='Viridis',
            # Choose a colorscale (e.g., 'Viridis', 'Jet', 'Plasma', 'Portland', 'Hot', 'Greys', etc.)
            showscale=True,  # <--- IMPORTANT: Set to True to display the colorbar
            colorbar=dict(  # <--- Optional: Customize the colorbar
                title='Elevation (deg)',  # Title for your colorbar
                x=1.02,  # Position of the colorbar (1.0 is right edge of plot)
                y=0.5,  # Vertical position (0.5 is center)
                len=0.7,  # Length of the colorbar (as a fraction of plot height)
                thickness=20  # Thickness of the colorbar
            )
        )
    ))
    # Update layout
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Multipath',
        title='Multipath Twr old',
        legend_title='twr old',
        template='plotly_white'
    )
    fig.show()
    fig.write_html(fr'{path}\LaeTwrOld_L1_{sv_galileo}_multipath.html')

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=galileo['Elevation_grn'][:1000],
        y=galileo['MP1_grn2'][:1000],
        mode='markers',
        name=f'SV Type: {sv_galileo}',  # Name for the legend (this will be for the trace itself, not the colorbar)
        marker=dict(
            size=8,
            opacity=0.7,
            color=galileo['S1_grn'][:1000],  # Data that determines the color
            colorscale='Viridis',
            # Choose a colorscale (e.g., 'Viridis', 'Jet', 'Plasma', 'Portland', 'Hot', 'Greys', etc.)
            showscale=True,  # <--- IMPORTANT: Set to True to display the colorbar
            colorbar=dict(  # <--- Optional: Customize the colorbar
                title='signal',  # Title for your colorbar
                x=1.02,  # Position of the colorbar (1.0 is right edge of plot)
                y=0.5,  # Vertical position (0.5 is center)
                len=0.7,  # Length of the colorbar (as a fraction of plot height)
                thickness=20  # Thickness of the colorbar
            )
        )
    ))
    # Update layout
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Multipath',
        legend_title='twr new',
        title='Multipath Twr New',
        template='plotly_white'
    )
    fig.show()
    fig.write_html(fr'{path}\LaeTwrNew_L1_{sv_galileo}_multipath.html')


def compare_timeseries():
    path2 = r'S:\group\rsws_gnss\VOD_timeseries_live\Lae2\Lae2_2025_VOD_timeseries_bl15days_noR.nc'
    path = r'S:\group\rsws_gnss\VOD_timeseries_live\Laeg\Laeg_2025_VOD_timeseries_bl15days_noR.nc'
    data_old = xr.open_mfdataset(path2).to_dataframe().dropna(how='all')
    data_new = xr.open_mfdataset(path).to_dataframe().dropna(how='all')
    from gnssvod.analysis.vod_plots import hemi_dif_plot2, plot_hemisphere

    data_old_aug = data_old[data_old.index.get_level_values('Epoch').dayofyear >= 205]
    data_new_aug = data_new[data_new.index.get_level_values('Epoch').dayofyear >= 205]

    hemi = gv.hemibuild(2)
    patches = hemi.patches()

    common_indices = data_new_aug.index.intersection(data_old_aug.index)
    data_old_aug = data_old_aug.loc[common_indices]
    data_new_aug = data_new_aug.loc[common_indices]

    vod_avg1 = data_old_aug.groupby(['CellID']).mean(['mean', 'std', 'count'])
    vod_avg2 = data_new_aug.groupby(['CellID']).mean(['mean', 'std', 'count'])
    common_indices = vod_avg1.index.intersection(vod_avg2.index)
    vod_avg1 = vod_avg1.loc[common_indices]
    vod_avg2 = vod_avg2.loc[common_indices]
    vod_avg = vod_avg1.copy()
    vod_avg["VOD"][:] = vod_avg1["VOD"][:] - vod_avg2["VOD"][:]
    plot_hemisphere(vod_avg, "VOD", patches, [-0.1, 0.2], r"S:\data\flox\Output\web_report\gnss\LaeComp",
                    title="LaeComp", ele_lim=60)

    idat_old = data_old_aug.dropna().groupby(
        [pd.Grouper(freq=f'30min', level='Epoch'), "SV"]).mean().reset_index()
    idat2_new = data_new_aug.dropna().groupby(
        [pd.Grouper(freq=f'30min', level='Epoch'), "SV"]).mean().reset_index()

    first_letters = [sv[0] for sv in idat_old['SV'].values]
    idat_old['SV_type'] = first_letters
    first_letters = [sv[0] for sv in idat2_new['SV'].values]
    idat2_new['SV_type'] = first_letters
    sv_galileo = "E"
    idat2_new = idat2_new[idat2_new['SV_type'] == sv_galileo]
    idat_old = idat_old[idat_old['SV_type'] == sv_galileo]

    idat_old = idat_old.drop(columns=["SV", "SV_type"]).dropna().groupby(
        [pd.Grouper(freq=f'2h', key='Epoch')]).mean().reset_index()
    idat2_new = idat2_new.drop(columns=["SV", "SV_type"]).dropna().groupby(
        [pd.Grouper(freq=f'2h', key='Epoch')]).mean().reset_index()

    # ALL
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=idat_old['Epoch'],  # X-axis: The 'Epoch' column
        y=idat_old['VOD'],  # Y-axis: The 'S1_reldif' column
        mode='lines+markers',  # Plot points instead of lines
        name=f'old'  # Name for the legend
    ))
    fig.add_trace(go.Scatter(
        x=idat2_new['Epoch'],  # X-axis: The 'Epoch' column
        y=idat2_new['VOD'],  # Y-axis: The 'S1_reldif' column
        mode='lines+markers',  # Plot points instead of lines
        name=f'new'  # Name for the legend
    ))

    # Update layout
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='GNSS-VOD (L1)',
        legend_title='Measurements',
        template='plotly_white'
    )
    fig.show()


def compare_vod_annual(product_paths, year, baseline, out_path, max_elevation=90, ignore_glosas=False):
    station_names = list(product_paths.keys())

    print(f"Plot vod comaprison at sites {station_names} at {out_path}")
    # houragg = "1h"
    exclude = []
    if ignore_glosas:
        filename_ext = f"_noR"
        exclude.append("")
    else:
        filename_ext = f""
        exclude.append("_noR")
    if max_elevation < 90:
        filename_ext = f"_maxEle{max_elevation}{filename_ext}"
        exclude.append("")
    else:
        exclude.append(f"_maxEle60")

    sv_galileo = "E"
    fig = go.Figure()

    datas_e = []
    datas = []

    for station_name in station_names:
        product_path = product_paths[station_name]

        print(f"Filename: {filename_ext}, exclude: {exclude} at {product_path}")
        # Load and filter input files
        files_product = index_data_files(product_path, station_name, "annual")
        files_product = filter_index_files(files_product, year=year, baseline=baseline, text=filename_ext, notincluded=exclude[0])
        files_product = filter_index_files(files_product, year=year, baseline=baseline, text=filename_ext, notincluded=exclude[1])
        print(f"files: {files_product['File'][0]}")
        ds_product = xr.open_dataset(files_product["File"][0]).to_dataframe()
        ds_product = ds_product[ds_product.index.year == int(year)]
        dat = ds_product.dropna().groupby(
            [pd.Grouper(freq=f'15min', level='Epoch'), "SV"]).mean().reset_index()
        first_letters = [sv[0] for sv in dat['SV'].values]
        dat['SV_type'] = first_letters
        dat_e = dat[dat['SV_type'] == sv_galileo]

        dat_e = dat_e.drop(columns=["SV", "SV_type"]).dropna().groupby(
            [pd.Grouper(freq=f'24h', key='Epoch')]).mean().reset_index()
        dat = dat.drop(columns=["SV", "SV_type"]).dropna().groupby(
            [pd.Grouper(freq=f'24h', key='Epoch')]).mean().reset_index()
        datas.append(dat)
        datas_e.append(dat_e)


        fig.add_trace(go.Scatter(
            x=dat['Epoch'],  # X-axis: The 'Epoch' column
            y=dat['VOD'],  # Y-axis: The 'S1_reldif' column
            mode='lines+markers',  # Plot points instead of lines
            name=f'{station_name} sat'  # Name for the legend
        ))
        fig.add_trace(go.Scatter(
            x=dat_e['Epoch'],  # X-axis: The 'Epoch' column
            y=dat_e['VOD'],  # Y-axis: The 'S1_reldif' column
            mode='lines+markers',  # Plot points instead of lines
            name=f'{station_name} galileo'  # Name for the legend
        ))

    # Update layout
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='GNSS-VOD (L1)',
        legend_title='Measurements',
        template='plotly_white'
    )
    fig.write_html(fr'{out_path}\{station_names[0]}_site_comparison.html')
    print(f"Plot vod comparison at site {station_names} at {out_path}")


def compare_vod():
    path2 = r'S:\group\rsws_gnss\VOD_t2\Lae2'
    path = r'S:\group\rsws_gnss\VOD_t2\Laeg'
    path3 = r'S:\group\rsws_gnss\VOD_t2\wrong_pair'
    extension = f"\*.nc"  # Linux extension
    filepattern = path + extension
    filepattern2 = path2 + extension
    extension = f"*.nc"  # Linux extension

    # different
    # pairing = ('Laeg2_Twr', 'Lae3T')
    # bands = ['S1']  # , "S2"]
    # All files and directories ending with .txt and that don't begin with a dot:
    files = sorted(glob.glob(filepattern))
    files2 = sorted(glob.glob(filepattern2))
    # Open paired files
    data_old = xr.open_mfdataset(files[51:]).to_dataframe().dropna(how='all')
    data_new = xr.open_mfdataset(files2).to_dataframe().dropna(how='all')

    t = 0
    idat_old = data_old.dropna().groupby(
        [pd.Grouper(freq=f'15min', level='Epoch'), "SV"]).mean().reset_index()
    idat2_new = data_new.dropna().groupby(
        [pd.Grouper(freq=f'15min', level='Epoch'), "SV"]).mean().reset_index()
    first_letters = [sv[0] for sv in idat_old['SV'].values]
    idat_old['SV_type'] = first_letters
    first_letters = [sv[0] for sv in idat2_new['SV'].values]
    idat2_new['SV_type'] = first_letters

    sv_galileo = "E"
    idat2_new = idat2_new[idat2_new['SV_type'] == sv_galileo]
    idat_old = idat_old[idat_old['SV_type'] == sv_galileo]

    idat_old = idat_old.drop(columns=["SV", "SV_type"]).dropna().groupby(
        [pd.Grouper(freq=f'1h', key='Epoch')]).mean().reset_index()
    idat2_new = idat2_new.drop(columns=["SV", "SV_type"]).dropna().groupby(
        [pd.Grouper(freq=f'1h', key='Epoch')]).mean().reset_index()

    # ALL
    fig = go.Figure()
    # unique_sv_types = idat3['SV_type'].unique()
    # # Iterate over each unique SV_type to create a separate trace
    # for sv_type_val in unique_sv_types:
    #     # Filter the DataFrame for the current SV_type
    #     df_filtered = idat3[idat3['SV_type'] == sv_type_val]
    fig.add_trace(go.Scatter(
        x=idat2_new['Epoch'],  # X-axis: The 'Epoch' column
        y=idat2_new['VOD'],  # Y-axis: The 'S1_reldif' column
        mode='lines+markers',  # Plot points instead of lines
        name=f'new'  # Name for the legend
    ))
    fig.add_trace(go.Scatter(
        x=idat_old['Epoch'],  # X-axis: The 'Epoch' column
        y=idat_old['VOD'],  # Y-axis: The 'S1_reldif' column
        mode='lines+markers',  # Plot points instead of lines
        name=f'old'  # Name for the legend
    ))

    # Update layout
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='GNSS-VOD (L1)',
        legend_title='Measurements',
        template='plotly_white'
    )
    fig.show()


def laeg_wrong_vods():
    path2 = r'S:\group\rsws_gnss\VOD_t2\Lae2'
    path = r'S:\group\rsws_gnss\VOD_t2\Laeg'
    path3 = r'S:\group\rsws_gnss\VOD_t2\wrong_pair'


    extension = f"\*.nc"  # Linux extension
    filepattern = path + extension
    filepattern2 = path2 + extension
    extension = f"*.nc"  # Linux extension
    filepattern3 = rf"{path3}\vod_Laeg{extension}"
    filepattern4 =rf"{path3}\vod_Lae2{extension}"

    # different
    # pairing = ('Laeg2_Twr', 'Lae3T')
    # bands = ['S1']  # , "S2"]

    # All files and directories ending with .txt and that don't begin with a dot:
    files = glob.glob(filepattern)
    files2 = glob.glob(filepattern2)
    files3 = glob.glob(filepattern3)
    files4 = glob.glob(filepattern4)

    # Open paired files
    data_old = xr.open_mfdataset(files[51:]).to_dataframe().dropna(how='all')
    data_new = xr.open_mfdataset(files2).to_dataframe().dropna(how='all')
    data_wrong1 = xr.open_mfdataset(files3).to_dataframe().dropna(how='all')
    data_wrong2 = xr.open_mfdataset(files4).to_dataframe().dropna(how='all')

    t=0

    idat_old = data_old.dropna().groupby(
        [pd.Grouper(freq=f'15min', level='Epoch'), "SV"]).mean().reset_index()
    idat2_new = data_new.dropna().groupby(
        [pd.Grouper(freq=f'15min', level='Epoch'), "SV"]).mean().reset_index()
    idat3_wrong = data_wrong1.dropna().groupby(
        [pd.Grouper(freq=f'15min', level='Epoch'), "SV"]).mean().reset_index()
    idat4_wrong = data_wrong2.dropna().groupby(
        [pd.Grouper(freq=f'15min', level='Epoch'), "SV"]).mean().reset_index()

    first_letters = [sv[0] for sv in idat_old['SV'].values]
    idat_old['SV_type'] = first_letters
    first_letters = [sv[0] for sv in idat2_new['SV'].values]
    idat2_new['SV_type'] = first_letters
    first_letters = [sv[0] for sv in idat3_wrong['SV'].values]
    idat3_wrong['SV_type'] = first_letters
    first_letters = [sv[0] for sv in idat4_wrong['SV'].values]
    idat4_wrong['SV_type'] = first_letters


    sv_galileo = "E"
    idat2_new = idat2_new[idat2_new['SV_type'] == sv_galileo]
    idat_old = idat_old[idat_old['SV_type'] == sv_galileo]
    idat3_wrong = idat3_wrong[idat3_wrong['SV_type'] == sv_galileo]
    idat4_wrong = idat4_wrong[idat4_wrong['SV_type'] == sv_galileo]

    idat_old = idat_old.drop(columns=["SV", "SV_type"]).dropna().groupby(
        [pd.Grouper(freq=f'1h', key='Epoch')]).mean().reset_index()
    idat2_new = idat2_new.drop(columns=["SV", "SV_type"]).dropna().groupby(
        [pd.Grouper(freq=f'1h', key='Epoch')]).mean().reset_index()
    idat3_wrong = idat3_wrong.drop(columns=["SV", "SV_type"]).dropna().groupby(
        [pd.Grouper(freq=f'1h', key='Epoch')]).mean().reset_index()
    idat4_wrong = idat4_wrong.drop(columns=["SV", "SV_type"]).dropna().groupby(
        [pd.Grouper(freq=f'1h', key='Epoch')]).mean().reset_index()

    # ALL
    fig = go.Figure()
    # unique_sv_types = idat3['SV_type'].unique()
    # # Iterate over each unique SV_type to create a separate trace
    # for sv_type_val in unique_sv_types:
    #     # Filter the DataFrame for the current SV_type
    #     df_filtered = idat3[idat3['SV_type'] == sv_type_val]
    fig.add_trace(go.Scatter(
        x=idat2_new['Epoch'],  # X-axis: The 'Epoch' column
        y=idat2_new['VOD'],  # Y-axis: The 'S1_reldif' column
        mode='lines+markers',  # Plot points instead of lines
        name=f'new'  # Name for the legend
    ))
    fig.add_trace(go.Scatter(
        x=idat_old['Epoch'],  # X-axis: The 'Epoch' column
        y=idat_old['VOD'],  # Y-axis: The 'S1_reldif' column
        mode='lines+markers',  # Plot points instead of lines
        name=f'old'  # Name for the legend
    ))
    fig.add_trace(go.Scatter(
        x=idat3_wrong['Epoch'],  # X-axis: The 'Epoch' column
        y=idat3_wrong['VOD'],  # Y-axis: The 'S1_reldif' column
        mode='lines+markers',  # Plot points instead of lines
        name=f'wrong 1'  # Name for the legend
    ))
    fig.add_trace(go.Scatter(
        x=idat4_wrong['Epoch'],  # X-axis: The 'Epoch' column
        y=idat4_wrong['VOD'],  # Y-axis: The 'S1_reldif' column
        mode='lines+markers',  # Plot points instead of lines
        name=f'wrong 2',  # Name for the legend
        marker=dict(size=8, opacity=0.7)  # Optional: Customize marker size and transparency
    ))
    # Update layout
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='GNSS-VOD (L1)',
        legend_title='Measurements',
        template='plotly_white'
    )
    fig.show()



def do_comparison(filepattern,pairings,bands, do_vod=True, do_twr=True, do_grnd=True):
    station_name = list(filepattern.keys())[0]
    print("Comparison called")
    print(f"Site: {station_name}\n do vod: {do_vod}\n do twr: {do_twr}\n do grnd: {do_grnd}")
    test_sat_const()
    if False: #do_vod:
        compare_vod()
    if do_twr:
        compare_twr(filepattern,pairings,bands,)


def compare_twr(filepattern,pairings,bands,path=r"C:\Users\michni\Documents\graphs"):

    station_name = list(filepattern.keys())[0]
    filepattern = filepattern[station_name]
    #files = get_filelist({station_name: filepattern})
    files=glob.glob(filepattern)


    #path=r'S:\group\rsws_gnss\Paired_t2\LaeCompare'
    # extension = f"\*.nc" # Linux extension
    # filepattern = path + extension
    # # different
    # pairing = ('Laeg2_Twr', 'Lae3T')
    bands = ['S1']#, "S2"]
    ref = pairings[station_name][0]
    grnd = pairings[station_name][1]

    # All files and directories ending with .txt and that don't begin with a dot:
    #files=glob.glob(filepattern)

    # Open paired files
    data = xr.open_mfdataset(files[10:40]).to_dataframe().dropna(how='all')

    iref = data.xs(ref, level='Station')
    igrn = data.xs(grnd, level='Station')
    idat = iref.merge(igrn, on=['Epoch', 'SV'], suffixes=['_ref', '_grn'])


    ivars = np.intersect1d(data.columns.to_list(), bands)
    for ivar in ivars:
        irefname = f"{ivar}_ref"
        igrnname = f"{ivar}_grn"
        ielename = f"Elevation_grn"
        # VOD calculation formula
        idat[f'{ivar}_vod'] = -np.log(np.power(10, (idat[igrnname] - idat[irefname]) / 10)) \
                     * np.cos(np.deg2rad(90 - idat[ielename]))
        idat[f'{ivar}_dif'] = idat[igrnname] - idat[irefname]
        idat[f'{ivar}_reldif'] = (idat[igrnname] - idat[irefname])/idat[irefname]*100


    idat2 = idat#[idat["Elevation_ref"]]
    # idat2 = idat2[idat2["S1_ref"]>10]
    # idat2 = idat2[idat2["S1_grn"]>10]
    idat2 = idat2.dropna().groupby(
                [pd.Grouper(freq=f'1min', level='Epoch'), 'SV']).mean()
    idat2 = idat2.reset_index('SV')

    first_letters = [sv[0] for sv in idat2['SV'].values]
    idat2['SV_type'] = first_letters

    idat2 = idat2.reset_index()#.set_index(['Epoch', 'SV_type'])

    # Galileo REL DIF:
    def galileo_plots():
        sv_galileo = "E"
        # Iterate over each unique SV_type to create a separate trace
        # Filter the DataFrame for the current SV_type
        df_filtered = idat2[idat2['SV_type'] == sv_galileo]

        # Relative difference (NEW-OLD) over time
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_filtered['Epoch'],
            y=df_filtered['S1_reldif'],
            mode='markers',
            name=f'SV Type: {sv_galileo}',  # Name for the legend (this will be for the trace itself, not the colorbar)
            marker=dict(
                size=8,
                opacity=0.7,
                color=df_filtered['Elevation_ref'],  # Data that determines the color
                colorscale='Viridis',
                # Choose a colorscale (e.g., 'Viridis', 'Jet', 'Plasma', 'Portland', 'Hot', 'Greys', etc.)
                showscale=True,  # <--- IMPORTANT: Set to True to display the colorbar
                colorbar=dict(  # <--- Optional: Customize the colorbar
                    title='Elevation (deg)',  # Title for your colorbar
                    x=1.02,  # Position of the colorbar (1.0 is right edge of plot)
                    y=0.5,  # Vertical position (0.5 is center)
                    len=0.7,  # Length of the colorbar (as a fraction of plot height)
                    thickness=20  # Thickness of the colorbar
                )
            )
        ))
        # Update layout
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='rel-dif S1',
            legend_title='Measurements',
            template='plotly_white',
            title="Relative difference (NEW-OLD) over time"
        )
        #fig.show()
        fig.write_html(fr'{path}\{station_name}_reldif_time_toc_galileo.html')

        # fig2: elevation - galileo
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_filtered['Elevation_ref'],
            y=df_filtered['S1_reldif'],
            mode='markers',
            name=f'SV Type: {sv_galileo}',  # Name for the legend (this will be for the trace itself, not the colorbar)
            marker=dict(
                size=8,
                opacity=0.7,
                color=df_filtered['Azimuth_ref'],  # Data that determines the color
                colorscale='Viridis',
                # Choose a colorscale (e.g., 'Viridis', 'Jet', 'Plasma', 'Portland', 'Hot', 'Greys', etc.)
                showscale=True,  # <--- IMPORTANT: Set to True to display the colorbar
                colorbar=dict(  # <--- Optional: Customize the colorbar
                    title='Azimuth_ref (deg)',  # Title for your colorbar
                    x=1.02,  # Position of the colorbar (1.0 is right edge of plot)
                    y=0.5,  # Vertical position (0.5 is center)
                    len=0.7,  # Length of the colorbar (as a fraction of plot height)
                    thickness=20  # Thickness of the colorbar
                )
            )
        ))
        # Update layout
        fig.update_layout(
            xaxis_title='Elevation (deg)',
            yaxis_title='Relative difference (NEW-OLD) over Elevation',
            legend_title='Measurements',
            template='plotly_white',
            title="Relative difference (NEW-OLD) over Elevation"
        )
        # fig.show()
        fig.write_html(fr'{path}\{station_name}_reldif_ele_toc_galileo.html')

        # fig 3: galileo: azimuth
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_filtered['Azimuth_ref'],
            y=df_filtered['S1_reldif'],
            mode='markers',
            name=f'SV Type: {sv_galileo}',  # Name for the legend (this will be for the trace itself, not the colorbar)
            marker=dict(
                size=8,
                opacity=0.7,
                color=df_filtered['Elevation_ref'],  # Data that determines the color
                colorscale='Viridis',
                # Choose a colorscale (e.g., 'Viridis', 'Jet', 'Plasma', 'Portland', 'Hot', 'Greys', etc.)
                showscale=True,  # <--- IMPORTANT: Set to True to display the colorbar
                colorbar=dict(  # <--- Optional: Customize the colorbar
                    title='Elevation_ref (deg)',  # Title for your colorbar
                    x=1.02,  # Position of the colorbar (1.0 is right edge of plot)
                    y=0.5,  # Vertical position (0.5 is center)
                    len=0.7,  # Length of the colorbar (as a fraction of plot height)
                    thickness=20  # Thickness of the colorbar
                )
            )
        ))
        # Update layout
        fig.update_layout(
            xaxis_title='Azimuth',
            yaxis_title='Relative difference (L1)',
            legend_title='Measurements',
            template='plotly_white',
            title="Relative difference (NEW-OLD) over Azimuth"
        )
        # fig.show()
        fig.write_html(fr'{path}\{station_name}_reldif_azi_toc_galileo.html')

    def galileo_correction():
        # --- 2. Define the Correction Function (predicting S2 from S1 and angles) ---
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.pipeline import make_pipeline
        from sklearn.metrics import mean_squared_error, r2_score

        # Pre-correction
        df_gnss = idat2[idat2["SV_type"]=="E"].copy()
        hemi = gv.hemibuild(2)
        df_gnss = hemi.add_CellID(df_gnss, aziname='Azimuth_ref', elename='Elevation_ref').drop(
            columns=['Azimuth_ref', 'Elevation_ref'])
        plot_dyn_hemispherical(df_gnss, path, vod_name="S1_ref",
                               title=f'S1 new', z_lim=[35, 48])
        plot_dyn_hemispherical(df_gnss, path, vod_name="S1_grn", title=f'S1 old', z_lim=[35, 48])

        def apply_elevation_correction(
                df: pd.DataFrame,
                signal_col: str,
                elevation_col: str,
                degree: int = 2  # Degree for polynomial features to model elevation dependency
        ) -> pd.Series:
            """
            Corrects a signal time series for its dependency on elevation angle.

            Args:
                df (pd.DataFrame): DataFrame containing the signal and elevation columns.
                signal_col (str): Name of the column for the original signal (e.g., 'Signal').
                elevation_col (str): Name of the column for the elevation angles (e.g., 'Elevation').
                degree (int): Degree of polynomial features to include in the regression model
                              to capture the elevation dependency.

            Returns:
                pd.Series: The corrected signal series.
            """
            # Prepare features (Elevation) and target (Signal) for the model
            X = df[[elevation_col]]  # Features must be 2D array-like
            y = df[signal_col]

            # Create a pipeline with PolynomialFeatures and LinearRegression
            # This model learns how the signal typically changes with elevation.
            model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())

            # Train the model on the data
            model.fit(X, y)

            # Predict the signal's expected value based on its elevation
            predicted_signal_from_elevation = model.predict(X)

            # Calculate the mean of the original signal
            mean_original_signal = y.mean()

            # Apply the correction:
            # Corrected = Original - (Predicted from Elevation) + (Mean Original Signal)
            # This removes the elevation-dependent component and shifts the signal to its mean.
            corrected_signal = y - predicted_signal_from_elevation + mean_original_signal

            return corrected_signal, predicted_signal_from_elevation, model  # Also return predicted and model for plotting/analysis

        # --- 3. Apply satellite elevation Correction ---
        # New sensor
        df_gnss['S1_ref_cor'], df_gnss['S1_ref_pred'], elevation_model = \
            apply_elevation_correction(
                df=df_gnss,
                signal_col='S1_ref',
                elevation_col='Elevation_grn',
                degree=5  # Using a quadratic model for elevation dependency
            )
        # old sensor
        df_gnss['S1_grn_cor'], df_gnss['S1_grn_pred'], elevation_model_grn = \
            apply_elevation_correction(
                df=df_gnss,
                signal_col='S1_grn',
                elevation_col='Elevation_grn',
                degree=5  # Using a quadratic model for elevation dependency
            )

        # FIG: correction elevation
        fig_elev_dep = go.Figure()
        # Original Signal vs. Elevation
        fig_elev_dep.add_trace(go.Scatter(
            x=df_gnss['Elevation_grn'],
            y=df_gnss['S1_ref'],
            mode='markers',
            name='Original Signal vs. Elevation',
            marker=dict(size=6, opacity=0.7, color='blue')
        ))
        # Original Signal vs. Elevation
        fig_elev_dep.add_trace(go.Scatter(
            x=df_gnss['Elevation_grn'],
            y=df_gnss['S1_ref_cor'],
            mode='markers',
            name='Corrected Signal vs. Elevation',
            marker=dict(size=6, opacity=0.7, color='green')
        ))
        # Plot the learned model's prediction line
        elevation_range = np.linspace(df_gnss['Elevation_grn'].min(), df_gnss['Elevation_grn'].max(), 100).reshape(-1,                                                                                               1)
        predicted_signal_smooth = elevation_model.predict(elevation_range)
        fig_elev_dep.add_trace(go.Scatter(
            x=elevation_range.flatten(),
            y=predicted_signal_smooth,
            mode='lines',
            name='Modelled Elevation Dependency',
            line=dict(color='orange', width=3, dash='solid')
        ))
        fig_elev_dep.update_layout(
            xaxis_title='Elevation Angle (degrees)',
            yaxis_title='Signal Strength (dB-Hz)',
            legend_title='Data',
            template='plotly_white',
            title='Elevation Correction new antenna'
        )
        #fig_elev_dep.show()
        fig_elev_dep.write_html(fr'{path}\{station_name}_elecornew_toc_galileo.html')

        plot_dyn_hemispherical(df_gnss, path, vod_name="S1_ref_cor",
                               title=f'S1 new antenna cor', z_lim=[30, 45])
        plot_dyn_hemispherical(df_gnss, path, vod_name="S1_grn_cor",
                               title=f'S1 old antenna cor', z_lim=[30, 45])


        def create_correction_model(
            df: pd.DataFrame,
            val1_col: str,
            val2_col: str,
            elevation_col: str,
            azimuth_col: str,
            degree: int = 2 # Degree for polynomial features
        ) -> make_pipeline:
            """
            Creates and trains a regression model to predict val2 from val1, elevation, and azimuth.

            Args:
                df (pd.DataFrame): DataFrame containing the measurement and angle columns.
                val1_col (str): Name of the column for Sensor 1's measurement (e.g., 'S1_reldif').
                val2_col (str): Name of the column for Sensor 2's measurement (e.g., 'S2_reldif').
                elevation_col (str): Name of the column for the common elevation angles.
                azimuth_col (str): Name of the column for the common azimuth angles.
                degree (int): Degree of polynomial features to include in the regression model.

            Returns:
                sklearn.pipeline.Pipeline: The trained scikit-learn pipeline (model).
            """
            # Features (X): S1_reldif, Elevation, Azimuth
            X = df[[val1_col, elevation_col, azimuth_col]]
            # Target (y): S2_reldif
            y = df[val2_col]

            # Create a pipeline with PolynomialFeatures and LinearRegression
            # PolynomialFeatures adds interaction terms (e.g., val1*elev, elev*azimuth) and higher powers
            model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())

            # Train the model
            model.fit(X, y)

            return model

        # --- 3. Train the model and get predictions ---
        df_measurements=idat2.copy()
        # Create the correction model
        correction_model = create_correction_model(
            df=df_measurements,
            val1_col='S1_ref',
            val2_col='S1_grn',
            elevation_col='Elevation_grn',
            azimuth_col='Azimuth_grn',
            degree=8 # Using quadratic terms for a more flexible model
        )

        # Prepare the features for prediction using Sensor 1's data and the common angles
        X_predict = df_measurements[['S1_ref', 'Elevation_grn', 'Azimuth_grn']]

        # Get the predicted S2_reldif values from S1_reldif and angles
        df_measurements['S1_grn_predicted_from_ref'] = correction_model.predict(X_predict)

        # --- 4. Check the New Mismatch ---
        # The mismatch is the difference between the actual S2_reldif and the predicted S2_reldif
        df_measurements['Mismatch_S1_predicted_vs_actual'] = \
            (df_measurements['S1_grn_predicted_from_ref'] - df_measurements['S1_grn']) / df_measurements['S1_grn'] * 100
        df_measurements['Mismatch_S1'] = (df_measurements['S1_ref'] - df_measurements['S1_grn'])/df_measurements['S1_grn'] * 100

        # Calculate evaluation metrics
        mse = mean_squared_error(df_measurements['S1_grn'], df_measurements['S1_grn_predicted_from_ref'])
        rmse = np.sqrt(mse)
        r2 = r2_score(df_measurements['S1_grn'], df_measurements['S1_grn_predicted_from_ref'])

        # Calculate evaluation metrics old
        mse_old = mean_squared_error(df_measurements['S1_grn'], df_measurements['S1_ref'])
        rmse_old = np.sqrt(mse_old)
        r2_old = r2_score(df_measurements['S1_grn'], df_measurements['S1_ref'])

        print(f"--- Mismatch Analysis ---")
        print(f"Mean Squared Error (MSE) of S2_predicted vs S2_actual: {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE) of S2_predicted vs S2_actual: {rmse:.4f}")
        print(f"R-squared (R2) of S2_predicted vs S2_actual: {r2:.4f}")
        print("\n")

        # --- 5. Plotting Initial Data and Prediction ---
        #
        # fig = go.Figure()
        #
        # # # Original Measurement 1 (S1_reldif)
        # # fig.add_trace(go.Scatter(
        # #     x=df_measurements.index,
        # #     y=df_measurements['S1_ref'],
        # #     mode='markers',
        # #     name='S1_reldif (Initial)',
        # #     marker=dict(size=6, opacity=0.8, color='blue')
        # # ))
        # #
        # # # Original Measurement 2 (S2_reldif) - The target we are trying to predict
        # # fig.add_trace(go.Scatter(
        # #     x=df_measurements.index,
        # #     y=df_measurements['S1_grn'],
        # #     mode='markers',
        # #     name='S2_reldif (Actual Target)',
        # #     marker=dict(size=6, opacity=0.8, color='green')
        # # ))
        # #
        # # # Predicted S2_reldif from S1_reldif and angles
        # # fig.add_trace(go.Scatter(
        # #     x=df_measurements.index,
        # #     y=df_measurements['S1_grn_predicted_from_ref'],
        # #     mode='markers',
        # #     name='S2_reldif (Predicted from S1 & Angles)',
        # #     marker=dict(size=6, opacity=0.8, color='red')
        # # ))
        #
        # # Plot the mismatch (optional, but good for understanding error distribution)
        # fig.add_trace(go.Scatter(
        #     x=df_measurements['Mismatch_S1'],
        #     y=df_measurements['Mismatch_S1_predicted_vs_actual'],
        #     mode='markers',
        #     name='Prediction Mismatch (Predicted - Actual)',
        #     marker=dict(size=4, opacity=0.6, color='purple'),
        #     yaxis='y2' # Plot on a secondary y-axis if values are very different
        # ))
        #
        #
        # # Update layout
        # fig.update_layout(
        #     xaxis_title='Date',
        #     yaxis_title='Measurement Value',
        #     legend_title='Data Series',
        #     template='plotly_white',
        #     hovermode='x unified',
        #     title='Predicting S2_reldif from S1_reldif and Angles',
        #     yaxis=dict(title='Measurement Value', side='left'),
        #     yaxis2=dict(title='Mismatch', overlaying='y', side='right', showgrid=False) # Secondary Y-axis
        # )
        #
        # fig.show()
        # df_filtered = idat4[idat4['SV_type'] == sv_type_val]

        df_prediction = df_measurements.copy()
        hemi = gv.hemibuild(2)
        # from gnssvod.analysis.vod_dyn_plot import plot_dyn_hemispherical

        df_prediction = hemi.add_CellID(df_prediction, aziname='Azimuth_ref', elename='Elevation_ref').drop(
            columns=['Azimuth_ref', 'Elevation_ref'])
        plot_dyn_hemispherical(df_prediction, path, vod_name="Mismatch_S1_predicted_vs_actual",
                               title=f'S1 pred', z_lim=[-15, 15])
        plot_dyn_hemispherical(df_prediction, path, vod_name="Mismatch_S1", title=f'S1 rel', z_lim=[-15, 15])


    galileo_correction()
    galileo_plots()

    # import gnssvod.gnssvod.hemistats.hemistats as hemistats
    # import gnssvod.gnssvod.hemistats as hemista
    # from gnssvod.gnssvod.hemistats import Hemi
    # from gnssvod.hemistats.hemistats import *

    # Initialize hemispheric grid
    hemi = gv.hemibuild(2)
    idat3 = hemi.add_CellID(idat2, aziname='Azimuth_ref', elename='Elevation_ref').drop(
        columns=['Azimuth_ref', 'Elevation_ref'])

    # idat3 = idat3[["SV_type", 'S1_reldif', 'Epoch', "CellID"]].set_index('Epoch').dropna().groupby(
    #             [pd.Grouper(freq=f'2min', level='Epoch'), 'SV_type', "CellID"]).mean()
    # idat3=idat3.reset_index()


    # idat4 = idat3[["SV_type", 'S1_reldif', 'Epoch', "CellID"]].set_index('Epoch').dropna().groupby(
    #     [pd.Grouper(freq=f'7D', level='Epoch'), 'SV_type', "CellID"]).mean()
    # idat4 = idat4.reset_index()
    unique_sv_types = idat3['SV_type'].unique()

    for sv_type_val in unique_sv_types:
        # Filter the DataFrame for the current SV_type
        df_filtered = idat3[idat3['SV_type'] == sv_type_val]
        plot_dyn_hemispherical(df_filtered, path, vod_name="S1_reldif", title=f'{sv_type_val} S1_reldif', z_lim=[-10,15])


    # ALL
    fig = go.Figure()

    unique_sv_types = idat3['SV_type'].unique()
    # Iterate over each unique SV_type to create a separate trace
    for sv_type_val in unique_sv_types:
        # Filter the DataFrame for the current SV_type
        df_filtered = idat3[idat3['SV_type'] == sv_type_val]

        fig.add_trace(go.Scatter(
            x=df_filtered['Epoch'],        # X-axis: The 'Epoch' column
            y=df_filtered['S1_reldif'],    # Y-axis: The 'S1_reldif' column
            mode='markers',                # Plot points instead of lines
            name=f'SV Type: {sv_type_val}',# Name for the legend
            marker=dict(size=8, opacity=0.7) # Optional: Customize marker size and transparency
        ))
        # 2. --- Calculate the Trendline ---
        df_sorted = df_filtered.sort_values('Epoch')
        window_size = 11  # (Must be an odd number for center=True to align)
        df_sorted['Rolling_Avg'] = df_sorted['S1_reldif'].rolling(
            window=window_size,
            center=True,
            min_periods=1
        ).mean()
        # 3. --- Add the Rolling Average Trace ---
        fig.add_trace(go.Scatter(
            x=df_sorted['Epoch'],
            y=df_sorted['Rolling_Avg'],
            mode='lines',
            name=f'SV Type: {sv_type_val}',
            line=dict(color='green', width=2)
        ))
    # Update layout
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='GNSS-VOD (L1)',
        legend_title='Measurements',
        template='plotly_white'
    )
    #fig.show()
    fig.write_html(fr'{path}\{station_name}_s1reldif_toc_bysv.html')

    fig = go.Figure()
    # for id, data in eth_maindata.items():
    #     fig.add_trace(go.Scatter(
    #         x=idat["Time"],
    #         y=data["ValueNorm"],
    #         mode='lines',
    #         name=f'{id}'
    #     ))
    unique_sv_types = idat2['SV_type'].unique()
    # Iterate over each unique SV_type to create a separate trace
    for sv_type_val in unique_sv_types:
        # Filter the DataFrame for the current SV_type
        df_filtered = idat2[idat2['SV_type'] == sv_type_val]

        fig.add_trace(go.Scatter(
            x=df_filtered['Epoch'],        # X-axis: The 'Epoch' column
            y=df_filtered['S1_reldif'],    # Y-axis: The 'S1_reldif' column
            mode='markers',                # Plot points instead of lines
            name=f'SV Type: {sv_type_val}',# Name for the legend
            marker=dict(size=8, opacity=0.7) # Optional: Customize marker size and transparency
        ))
    # Update layout
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='GNSS-VOD (L1)',
        legend_title='Measurements',
        template='plotly_white'
    )
    fig.write_html(fr'{path}\{station_name}_s1reldif2_toc_bysv.html')



    idat3 = idat2[["SV_type", 'S1_reldif', 'Epoch']].set_index('Epoch').dropna().groupby(
                [pd.Grouper(freq=f'5min', level='Epoch'), 'SV_type']).mean()
    idat3=idat3.reset_index()

    fig = go.Figure()

    unique_sv_types = idat3['SV_type'].unique()
    # Iterate over each unique SV_type to create a separate trace
    for sv_type_val in unique_sv_types:
        # Filter the DataFrame for the current SV_type
        df_filtered = idat3[idat3['SV_type'] == sv_type_val]

        fig.add_trace(go.Scatter(
            x=df_filtered['Epoch'],        # X-axis: The 'Epoch' column
            y=df_filtered['S1_reldif'],    # Y-axis: The 'S1_reldif' column
            mode='markers',                # Plot points instead of lines
            name=f'SV Type: {sv_type_val}',# Name for the legend
            marker=dict(size=8, opacity=0.7) # Optional: Customize marker size and transparency
        ))
    # Update layout
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='GNSS-VOD (L1)',
        legend_title='Measurements',
        template='plotly_white'
    )
    fig.write_html(fr'{path}\{station_name}_s1reldif_toc_bysv.html')
    #fig.show()

    t=0


