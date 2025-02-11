import datetime
import os.path
import sys
from gnssvod.Gnss_site import Gnss_site
import pandas as pd
import configparser
import xarray as xr
import gnssvod.plot
import logging
import matplotlib.pyplot as plt
import numpy as np
from datetime import date


CONFIG_FILENAME = "config.ini"


def readConfigIni():
    config_obj = configparser.ConfigParser()
    config_obj.read(CONFIG_FILENAME)

    general = config_obj["General"]
    site_number = general["siteNumber"]

    sites = []

    for i in range(1, int(site_number)+1):
        section_name = str(i)
        site = Gnss_site(config_obj[f"Site{section_name}"])
        sites.append(site)

    return sites


def plot_it():
    # -----------------------------------------
    # Test plot
    # Load the data
    # vod_dav = xr.open_dataset(r'C:\Users\jkesselr\Documents\PhD\Data\Objective2\GNSS\vod_Dav_20220101_20220201.nc')
    files = ["vod_Dav_20220101_20220131.nc", "vod_Dav_20220201_20220228.nc", "vod_Dav_20220301_20220331.nc",
             "vod_Dav_20220401_20220430.nc", "vod_Dav_20220501_20220531.nc", "vod_Dav_20220601_20220630.nc",
             "vod_Dav_20220701_20220731.nc", "vod_Dav_20220801_20220822.nc", "vod_Dav_20220928_20220930.nc",
             "vod_Dav_20221001_20221031.nc", "vod_Dav_20221101_20221130.nc", "vod_Dav_20221201_20221231.nc"]
    path = r"Z:\group\rsws_gnss\VOD\Dav"
    # vod_dav = [xr.open_mfdataset(os.path.join(path, x)).VOD for x in files]

    # path = r"X:\rsws_gnss\VOD\Laeg"
    # f1 = "vod_Laeg_20230701_20230731.nc"
    # f2 = "vod_Laeg_20220701_20220731.nc"
    # # vod_dav1 = xr.open_dataset(os.path.join(path, f1))
    # # vod_dav2 = xr.open_dataset(os.path.join(path, f2))
    # vod_dav1.SV
    # vod_dav2.SV


    vods = []
    times = []
    for f in files:
        vod_dav = xr.open_dataset(os.path.join(path, f)).VOD
        galileos = [x.startswith('E') for x in vod_dav.SV.data]
        vod_poi = vod_dav[:, galileos]
        # galileos2 = [x.startswith('E') for x in vod_dav[1].SV.data]
        # vod_poi2 = vod_dav[0][:, galileos2]
        vod_daily = vod_poi.sortby('Epoch').resample(Epoch='1D').mean()
        dailyvod = np.nanmean(vod_daily, axis=1)
        times.append(vod_daily.Epoch.values)
        vods.append(dailyvod)

    plt.figure()
    for a in range(0, len(vods)):
        plt.plot(times[a], vods[a], color="blue")
    plt.ylabel('VOD')
    plt.xlabel('Time')
    plt.title('Davos VOD - daily average all E-satellites')
    plt.legend()
    plt.xticks(rotation=25)
    plt.savefig('Dav_VOD_Galileo13_hourly.png')
    plt.show()

    # Extract tower and ground variables
    dav_S1_tower = vod_dav['S1_ref']
    dav_S1_ground = vod_dav['S1_grn']
    # Plot the data
    plt.plot(vod_dav['Epoch'], dav_S1_tower.sel(SV='S36'), label='tower')
    plt.plot(vod_dav['Epoch'], dav_S1_ground.sel(SV='S36'), label='ground')
    plt.ylabel('SNR')
    plt.xlabel('Time')
    plt.title('Davos geostationary satellite')
    plt.legend()
    plt.show()
    plt.plot(vod_dav['Epoch'], vod_dav['VOD'].sel(SV='E13'))
    plt.ylabel('VOD')
    plt.xlabel('Time')
    plt.title('Davos VOD - E13')
    plt.legend()
    plt.xticks(rotation=25)
    plt.savefig('Dav_VOD_Galileo13.png')
    plt.show()
    # Resample to hourly values
    dav_S1_tower_hourly = dav_S1_tower.resample(Epoch='1H').mean()
    dav_S1_ground_hourly = dav_S1_ground.resample(Epoch='1H').mean()
    dav_vod_hourly = vod_dav.resample(Epoch='1H').mean()
    # Plot the data
    plt.plot(dav_S1_tower_hourly['Epoch'], dav_S1_tower_hourly.sel(SV='S36'), label='tower')
    plt.plot(dav_S1_ground_hourly['Epoch'], dav_S1_ground_hourly.sel(SV='S36'), label='ground')
    plt.ylabel('SNR')
    plt.xlabel('Time')
    plt.title('Davos geostationary satellite')
    plt.legend()
    plt.xticks(rotation=25)
    plt.savefig('Dav_stationary_hourly.png')
    plt.show()

    plt.plot(dav_S1_tower_hourly['Epoch'], dav_S1_tower_hourly.sel(SV='E13'), label='tower')
    plt.plot(dav_S1_ground_hourly['Epoch'], dav_S1_ground_hourly.sel(SV='E13'), label='ground')
    plt.ylabel('SNR')
    plt.xlabel('Time')
    plt.title('Davos - E13')
    plt.legend()
    plt.xticks(rotation=25)
    plt.savefig('Dav_Galileo13_hourly.png')
    plt.show()
    plt.plot(dav_vod_hourly['Epoch'], dav_vod_hourly['VOD'].sel(SV='E13'))
    plt.ylabel('VOD')
    plt.xlabel('Time')
    plt.title('Davos VOD - E13')
    plt.legend()
    plt.xticks(rotation=25)
    plt.savefig('Dav_VOD_Galileo13_hourly.png')
    plt.show()


if __name__ == '__main__':
    args = sys.argv[1:]
    print("Main GNSS ------")
    print(args)
    # Setup logging
    logging.basicConfig(filename='gnss.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Initialize
    sites = readConfigIni()
    # Program mode
    is_preprocessing = False
    is_pairing = False
    is_vod = False
    is_timeseries = False
    is_product = False
    is_plot = False
    # Program parameters
    is_tower = False
    is_ground = False
    site = None
    start_date = None
    is_autotime = False
    is_lastweek = False
    year = None

    # Load console
    for i, arg in enumerate(args):
        if arg == '-n' or arg == '-netcdf':
            is_preprocessing = True
            is_tower = args[i + 1] == 'twr'
        if arg == '-p' or arg == '-pair':
            is_pairing = True
        if arg == '-v' or arg == '-vod':
            is_vod = True
        if arg == '-s' or arg == '-site':
            site = args[i + 1]
        if arg == '-plot':
            is_plot = True
        if arg == '-product':
            is_product = True
        if arg == '-timeseries':
            is_timeseries = True
        if arg == '-dates':
            if args[i+1] == "auto":
                is_autotime = True
            elif args[i+1] == "recent":
                dt = datetime.datetime.now() - datetime.timedelta(days=7)
                start_date = pd.Timestamp(f'{dt} 00:00:00')
                if year is None:
                    year = dt.year
            else:
                start_date = pd.Timestamp(f'{args[i + 1]} 00:00:00')
        if arg == '-year':
            year = args[i + 1]

    gnss_site = None
    for s in sites:
        if s.short_name == site:
            gnss_site = s

    if gnss_site is None:
        print("No site selected")

    # Create netcdf from raw data
    if is_preprocessing:
        # Preprocess one instrument of the station data from the raw files and stores it as netcdf files for weekly data.
        if start_date is None:
            start_date = pd.Timestamp("2022-01-01")
        gnss_site.preprocess(is_tower, start_date, is_autotime)
    elif is_pairing:
        # Pairs the station data (Tower and Ground) for the given month from date
        end_date = pd.to_datetime(start_date) + pd.offsets.MonthBegin(1)
        timeperiod = pd.interval_range(start=start_date, end=end_date, freq='D')
        gnss_site.pairing(timeperiod)
    elif is_vod:
        if year is None:
            print("No year defined. Cancel process.")
            print("Define with: '-year 2021'")
        else:
            # Calculates the VOD from the paired station data (tower and ground) and stores it for the given year
            year_plus = str(int(year)+1)
            timeperiod = pd.interval_range(start=pd.Timestamp(f"{year}-01-01 00:00:00"),
                                           end=pd.Timestamp(f"{year_plus}-01-01 00:00:00"), freq='MS')
            gnss_site.calculate_vod(timeperiod)
    elif is_timeseries:
        if year is None:
            print("No year defined. Cancel process.")
            print("Define with: '-year 2021'")
        else:
            # Creates the vod times and baselines and saves them
            gnss_site.create_timeseries(year)
    elif is_product:
        # Creates the vod products and saves them
        gnss_site.create_product()
    elif is_plot:
        # If year is None, do whole time series plot, else specific year jan to dec.
        if year is None:
            print("No year defined. Do whole timeseries")
            # TODO: Do site specific startyear
            year = 2021
            enddate = pd.Timestamp(f"{date.today().strftime('%Y-%m-%d')} 00:00:00")
        else:
            year_plus = str(int(year) + 1)
            enddate = pd.Timestamp(f"{year_plus}-01-01 00:00:00")
        timeperiod = pd.interval_range(start=pd.Timestamp(f"{year}-01-01 00:00:00"),
                                       end=enddate, freq='MS')
        # Plot time series
        gnss_site.plot_timeseries(year)

    else:
        print("No mode")





