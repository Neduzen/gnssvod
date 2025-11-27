import datetime
import os.path
import numpy as np
import pytz
import xarray as xr
import pandas as pd
import random
import string
import plotly.graph_objects as go
import plotly.express as px
from matplotlib import cm, pyplot as plt
import math
import plotly.io as pio
import gnssvod as gv


def add_angles_by_cellid(vod_avg, hemi, patches):
    # # # Initialize hemispheric grid and patches
    # hemi = gv.hemibuild(2)
    # patches = hemi.patches()

    vod_avg['zenith'] = np.full_like(vod_avg['CellID'], np.nan, dtype=float)
    vod_avg['azimuth'] = np.full_like(vod_avg['CellID'], np.nan, dtype=float)

    for cell_id in vod_avg['CellID'].values:
        if cell_id in hemi.coords.index.values:
            vod_avg['zenith'].loc[cell_id] = hemi.coords.loc[cell_id, 'ele']
            vod_avg['azimuth'].loc[cell_id] = hemi.coords.loc[cell_id, 'azi']
    return vod_avg, patches


def plot_eth_var_comparison(vod):
    path = r"C:\Users\michni\Desktop\RSWS\Sites\Laegern\ETH_data_test"
    eth_files = {"Dendro": "DENDRO_Lae_2025-06-23.csv",
                 "LeafWet": "LEAF_WET_Lae_2025-06-23.csv",
                 "Prec": "PREC_Lae_2025-06-23.csv",
                 "SapFlow": "SAPFLOW_Lae_2025-06-23.csv",
                 "SoilWaterContent": "SWC_Lae_2025-06-23.csv",
                 "SoilWaterPotential": "SWP_Lae_2025-06-23.csv",
                 }
    vod_cell_highfreq = vod.groupby(["CellID", pd.Grouper(freq="1H", level='Epoch')]).mean()
    vod_cell_highfreq_mean = vod_cell_highfreq.groupby(pd.Grouper(freq="1H", level='Epoch')).mean()

    eth_data = {}
    eth_maindata = {}
    for id, fname in eth_files.items():
        pathy = os.path.join(path, fname)
        df = pd.read_csv(pathy)
        eth_data[id] = df.copy()

        columns_to_average = df.columns.drop('Time')
        df['Value'] = df[columns_to_average].mean(axis=1)
        result_df = df[['Time', 'Value']].copy()
        result_df["ValueNorm"] = (df["Value"] - df["Value"].min()) / (df["Value"].max() - df["Value"].min())
        eth_maindata[id] = result_df.copy()

    hempathy1 = r"S:\group\rsws_gnss\VOD_product_live\Laeg\Laeg_VOD_product_bl15_1h_maxEle60_noR.nc"
    vod25 = xr.open_dataset(hempathy1).to_dataframe()

    fig = go.Figure()
    for id, data in eth_maindata.items():
        fig.add_trace(go.Scatter(
            x=data["Time"],
            y=data["ValueNorm"],
            mode='lines',
            name=f'{id}'
        ))
    fig.add_trace(go.Scatter(
        x=vod_cell_highfreq_mean.index,
        y=vod_cell_highfreq_mean["VOD"],
        mode='lines',
        name=f'VOD'
    ))
    # Update layout
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='GNSS-VOD (L1)',
        legend_title='Measurements',
        template='plotly_white'
    )
    fig.show()



def features_plot(vod, vod_name):
    features = pd.read_csv("laeg_features.csv")

    fig = go.Figure()

    # features = pd.read_csv("laeg_features.csv")
    vod_features = {}
    vod_mean_features = {}
    for colname in features:
        cellid_vod = vod.index.get_level_values('CellID')
        mask = cellid_vod.isin(features[colname])
        filtered_vod = vod[mask]
        vod_features[colname] = filtered_vod
        high_freq = '8H'
        low_freq = '3D'
        tick_format = '%y-%m-%d: %H'
        vod_high_freq = filtered_vod.groupby(pd.Grouper(freq=high_freq, level='Epoch')).mean()
        vod_low_freq = vod_high_freq.rolling(low_freq, center=True).mean()

        vod_cell_highfreq = filtered_vod.groupby(["CellID", pd.Grouper(freq=high_freq, level='Epoch')]).mean()
        vod_cell_highfreq_mean = vod_cell_highfreq.groupby(pd.Grouper(freq=high_freq, level='Epoch')).mean()
        # vod_cell_lowfreq = vod_cell_highfreq.groupby([pd.Grouper(freq=low_freq, level='Epoch')]).mean()
        vod_cell_lowfreq_mean = vod_cell_highfreq_mean.rolling(low_freq, center=True).mean()

        window = vod_cell_lowfreq_mean['VOD_raw'].rolling(
            window='15D')  # rolling(window=int(24 * int(baseline) * (1/hour_frequency)), min_periods=1, center=True)
        vod_cell_lowfreq_mean['VOD_anom_corr_final'] = vod_cell_lowfreq_mean['VOD_anom'] + window.mean()

        # vod_filtered = vod[vod.index.isin(features[colname])]
        # vod_features[colname] = vod_filtered
        # vod_mean_features[colname] = vod_filtered.mean(axis=0)
        # col = ["grey", "darkgreen"]
        fig.add_trace(go.Scatter(
            x=vod_low_freq.index,
            y=vod_low_freq[vod_name],
            mode='lines',
            name=f'{colname}: {vod_name} {low_freq}'
        ))
        fig.add_trace(go.Scatter(
            x=vod_cell_lowfreq_mean.index,
            y=vod_cell_lowfreq_mean[vod_name],
            mode='lines',
            name=f'{colname}: {vod_name} good avg {low_freq}'
        ))
        fig.add_trace(go.Scatter(
            x=vod_cell_lowfreq_mean.index,
            y=vod_cell_lowfreq_mean["VOD_anom_corr_final"],
            mode='lines',
            name=f'{colname}: VOD_anom_corr_final {low_freq}'
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
    fig.show()



def plot_dyn_hemispherical(vod, out_path, vod_name="VOD_anom_corr", agg_limit=[None,None,None], title="", z_lim = [0, 3]):
    """
    Creates the monthly VOD hemispherical plot for a specific month and year and site.
    agg_limit = max_elev, min_tot, min_cell
    """
    if title == "":
        title = vod_name
    # # # Initialize hemispheric grid and patches
    elevation_angle = agg_limit[0]

    hemi = gv.hemibuild(2)
    patches = hemi.patches()
    vod_avg = vod.groupby(['CellID']).mean(['mean', 'std', 'count'])
    vod_avg = vod_avg.reset_index()

    # vod_avg["CountVals"] = vod["CountVals"].sum(dim='Epoch')
    # vod_avg["Cell_count"] = vod["CountVals"].count(dim='Epoch')

    # mask = (vod_avg[vod_name] > -2).compute()
    # vod_avg = vod_avg.where(mask, drop=True)
    # mask = (vod_avg["Cell_count"] > 0).compute()
    # vod_avg = vod_avg.where(mask, drop=True)
    # mask = (~vod_avg[vod_name].isnull()).compute()
    # vod_avg = vod_avg.where(mask, drop=True)     # Filter NaN cells
    # if agg_limit[1] is not None:
    #     vod_avg = vod_avg.where(vod_avg["CountVals"]>agg_limit[1], drop=True)
    # if agg_limit[2] is not None:
    #     vod_avg = vod_avg.where(vod_avg["Cell_count"]>agg_limit[2], drop=True)


    vod_avg, patches = add_angles_by_cellid(vod_avg, hemi, patches)
    # Filter low elevation cells
    if elevation_angle is not None:
        mask = (vod_avg["zenith"] > 90 - elevation_angle).compute()
        vod_avg = vod_avg.where(mask, drop=True)  # Filter NaN cells
    # z_lim = [0, 3]

    ivod_data = pd.Series(vod_avg[vod_name], name=vod_name, index=vod_avg["CellID"])
    # ivod_data = ivod_data[ivod_data.values > -1]
    ipatches = pd.concat([pd.Series(patches, name='Patches'), ivod_data], join='inner', axis=1)
    ipatches = ipatches[~np.isnan(ipatches[vod_name])]

    cmap = cm.get_cmap('RdBu')
    # cmap = cm.berlin
    norm = plt.Normalize(vmin=z_lim[0], vmax=z_lim[1])
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    plotly_colorscale = [[norm(v), f'rgb({int(c[0] * 255)}, {int(c[1] * 255)}, {int(c[2] * 255)})']
                         for v, c in zip(np.linspace(z_lim[0], z_lim[1], cmap.N), cmap(np.linspace(0, 1, cmap.N)))]

    fig = go.Figure()

    for index, row in ipatches.iterrows():
        rect = row['Patches']  # Get the Rectangle object
        x, y = rect.xy  # Bottom-left corner
        width = rect.get_width()
        height = rect.get_height()
        color_value = row[vod_name]

        # Calculate vertices (assuming x=theta, y=r)
        theta_deg = np.array([x, x + width, x + width, x]) * 180 / np.pi  # Convert to degrees
        r = np.array([y, y, y + height, y + height])

        fillcol = int(((color_value - z_lim[0]) / (z_lim[1] - z_lim[0])) * (len(plotly_colorscale) - 1))
        if fillcol < 0:
            fillcol = 0
        elif fillcol >= len(plotly_colorscale):
            fillcol = len(plotly_colorscale) - 1

        fig.add_trace(go.Scatterpolar(
            r=r,
            theta=theta_deg,
            mode='lines',
            fill='toself',
            line=dict(color='black', width=1),
            fillcolor=plotly_colorscale[fillcol][1],
            name=f'Value: {color_value:.2f}',
            hovertemplate=f'Theta: %{{theta:.1f}}°<br>R: %{{r:.1f}}<br>GNSS-VOD: {color_value:.2f}<extra></extra>',
            showlegend=False
        ))
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(
            size=0,
            color=[z_lim[0], z_lim[1]],  # Dummy color data to set the range
            colorscale=plotly_colorscale,
            colorbar=dict(
                title=title,
                thickness=20,
                xpad=10,
                ypad=10,
                lenmode='fraction',
                len=0.5,
                yanchor='bottom',
                y=0,
                tickvals=np.linspace(z_lim[0], z_lim[1], 5),
                ticktext=[f'{v:.1f}' for v in np.linspace(z_lim[0], z_lim[1], 5)],
            )
        ),
        showlegend=False
    ))
    angle_round = 90
    if elevation_angle is not None:
        angle_round = math.ceil(elevation_angle / 10) * 10
    fig.update_layout(
        polar=dict(
            bgcolor="black",
            radialaxis=dict(range=[0, angle_round], visible=True, tickvals=np.arange(0, angle_round + 1, 20),
                            ticktext=[f'{i}°' for i in np.arange(0, angle_round + 1, 20)],
                            tickfont=dict(color='grey')),
            angularaxis=dict(direction="clockwise", rotation=90, tickvals=np.arange(0, 360, 45),
                             ticktext=['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'],
                             tickfont=dict(color='grey')),
        ),
        width=900,
        height=800,
        coloraxis=dict(
            colorbar=dict(
                title=vod_name,
                thickness=20,
                xpad=10,
                ypad=10,
                lenmode='fraction',
                len=0.5,
                yanchor='bottom',
                y=0,
                tickvals=np.linspace(z_lim[0], z_lim[1], 5),
                ticktext=[f'{v:.1f}' for v in np.linspace(z_lim[0], z_lim[1], 5)]
            ),
            cmin=z_lim[0],
            cmax=z_lim[1],
            colorscale=plotly_colorscale
        ),
        # --- Background colors ---
        paper_bgcolor='black',  # Color of the entire paper area (outside the plot)
        plot_bgcolor='black',  # Color of the plotting area itself (inside the axes)
        # --- Font color for general layout elements (title, legend, etc.) ---
        font=dict(color='white'),
        # paper_bgcolor='white',
        # font=dict(color='black')
    )

    # Create file
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    random_number = ''.join(random.choices(string.digits, k=6))
    title_name = title.replace(" ", "_")
    file_out = os.path.join(out_path, f"gnss_hemispherical_plot_{title_name}.html") #_{current_date}_{random_number}.html")
    if out_path is not None or out_path != "":
        fig.write_html(file_out)
    else:
        fig.show()
