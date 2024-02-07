import sys
from gnssvod.Gnss_site import Gnss_site
from gnssvod.io.preprocess import (preprocess, get_filelist, pair_obs, calc_vod)
import pandas as pd
import configparser


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


if __name__ == '__main__':
    args = sys.argv[1:]
    print("Main GNSS ------")
    print(args)

    sites = readConfigIni()

    is_preprocessing = False
    is_pairing = False
    is_vod = False
    is_tower = False
    is_ground = False
    site = None
    start_date = None
    is_autotime = False

    # site="Laeg"
    # station="Twr"
    # mode="-n"

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
        if arg == '-dates':
            if args[i+1] == "auto":
                is_autotime = True
            else:
                start_date = pd.Timestamp(f'{args[i+1]} 00:00:00')

    gnss_site = None
    for s in sites:
        if s.short_name == site:
            gnss_site = s

    if gnss_site is None:
        print("No site selected")
    if start_date is None:
        start = pd.Timestamp("2022-01-01")

    # Create netcdf from raw data
    if is_preprocessing:
        # now = pd.Timestamp(datetime.datetime.now().strftime("%Y-%m-%d 00:00:00"))-datetime.timedelta(7)
        gnss_site.preprocess(is_tower, start_date, is_autotime)
    elif is_pairing:
        # start = pd.Timestamp("2022-01-01")
        timeperiod = pd.interval_range(start=start_date, periods=31, freq='D')
        gnss_site.pairing(timeperiod)
    elif is_vod:
        # start = pd.Timestamp("2022-01-01")
        timeperiod = pd.interval_range(start=start_date, periods=31, freq='D')
        gnss_site.calculate_vod(timeperiod)
    # if mode=="-n":
    #     #ReachLaeg1G
    #     #ReachLaeg2T
    #
    #     if site == "Lae":
    #         keepvars = ["S1", "S2"]
    #             #["S1C", "S1X", "S2C", 'Azimuth', 'Elevation']  # , "S2I", "S2X", "S7I", "S7X"]
    #         # ['S1', 'S2', 'Azimuth', 'Elevation']
    #         if station == "Twr":
    #             input_pattern = {'ReachLaeg2T': r'C:\Users\mniederb\Documents\run\gnss\in\Laeg2_Twr\*.zip'}
    #             outdir = {'ReachLaeg2T': r'C:\Users\mniederb\Documents\run\gnss\out\Lae\ReachLaeg2T'}
    #         else:
    #             input_pattern = {
    #                 'ReachLaeg1G': r'C:\Users\mniederb\Documents\run\gnss\in\Laeg1_Grnd\*.zip'}
    #             outdir = {'ReachLaeg1G': r'C:\Users\mniederb\Documents\run\gnss\out\Lae\ReachLaeg1G'}
    #             # input_pattern = {'Laeg2': r'C:\Users\mniederb\Documents\run\gnss\in\Laeg2_Twr'}
    #             # {'station1':'/path/to/files/of/station1/*O'}
    #     else:
    #         keepvars = ["S1", "S2"]#, 'Azimuth', 'Elevation']
    #         if station == "Twr":
    #             input_pattern = {'Dav_Twr': r'C:\Users\mniederb\Documents\run\gnss\in\Dav2_Twr\*.zip'}
    #             outdir = {'Dav_Twr': r'C:\Users\mniederb\Documents\run\gnss\out\Dav\Dav_Twr'}
    #         else:
    #             input_pattern = {
    #                 'Dav_Grnd': r'C:\Users\mniederb\Documents\run\gnss\in\Dav1_Grnd\*.zip'}
    #             outdir = {'Dav_Grnd': r'C:\Users\mniederb\Documents\run\gnss\out\Dav\Dav_Grnd'}
    #             # input_pattern = {'Laeg2': r'C:\Users\mniederb\Documents\run\gnss\in\Laeg2_Twr'}
    #             # {'station1':'/path/to/files/of/station1/*O'}
    #
    #     print(f"Preprocess of {input_pattern} to {outdir}")
    #     preprocess(input_pattern, True, interval="60s", outputdir=outdir, keepvars=keepvars)
    #
    # elif mode=="-p":
    #     #pairings = {'case1': ('station1', 'station2')}
    #     #filepattern = {'station1': '/path/to/files/of/station1/*.nc',
    #     #               'station2': '/path/to/files/of/station2/*.nc'}
    #     timeperiod = pd.interval_range(start=pd.Timestamp('1/1/2023'), periods=31, freq='D')
    #
    #     if site == "Lae":
    #         pairings = {'Lae': ('ReachLaeg2T', 'ReachLaeg1G')}
    #         filepattern = {'ReachLaeg2T': r'C:\Users\mniederb\Documents\run\gnss\out\Lae\ReachLaeg2T\*.nc',
    #                        'ReachLaeg1G': r'C:\Users\mniederb\Documents\run\gnss\out\Lae\ReachLaeg1G\*.nc'}
    #         outputdir = {'Lae': r'C:\Users\mniederb\Documents\run\gnss\out\Lae\paired'}
    #     else:
    #         pairings = {'Dav': ('Dav2_Twr', 'Dav1_Grnd')}
    #         filepattern = {'Dav2_Twr': r'C:\Users\mniederb\Documents\run\gnss\out\Dav\Dav_Twr\*.nc',
    #                        'Dav1_Grnd': r'C:\Users\mniederb\Documents\run\gnss\out\Dav\Dav_Grnd\*.nc'}
    #         outputdir = {'Dav': r'C:\Users\mniederb\Documents\run\gnss\out\Dav\paired'}
    #
    #     print("Pairing: " + str(pairings))
    #     pair_obs(filepattern, pairings, timeperiod, outputdir=outputdir)
    # elif mode == "-v":
    #     if site=="Dav":
    #         filepattern = {'Dav': r'C:\Users\mniederb\Documents\run\gnss\out\Dav\paired\*.nc'} #'case2': '/path/to/files/of/case2/*.nc'}
    #         pairing = {'Dav': ('S1_ref', 'S1_grn', 'Elevation_grn')} #,'VOD2': ('S2C_ref', 'S2C_grn', 'Elevation_grn')}
    #         outputdir = {'Dav': r'C:\Users\mniederb\Documents\run\gnss\out\Dav\vod'}
    #     else:
    #         filepattern = {
    #             'Lae': r'C:\Users\mniederb\Documents\run\gnss\out\Lae\paired\*.nc'}
    #         pairing = {
    #             'Lae': ('S1C_ref', 'S1C_grn', 'Elevation_grn')}
    #         outputdir = {'Lae': r'C:\Users\mniederb\Documents\run\gnss\out\Lae\vod'}
    #
    #     print(f"vod: {filepattern}")
    #     result = calc_vod(filepattern, pairing, outputdir)
    #     print(result)
    else:
        print("No mode")





