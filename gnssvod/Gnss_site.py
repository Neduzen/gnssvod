import configparser
import datetime
import logging
import os
import sys
import datetime as datetime
from gnssvod.io.preprocess import (preprocess, get_filelist, gather_stations)
import pandas as pd
from gnssvod.analysis.vod_calc import process_vod
import gnssvod.analysis.vod_timeseries as vod_timeseries
import gnssvod.analysis.vod_plots as vod_plots


class Gnss_site:
    def __init__(self, site):
        self.config_parser = site
        self.name = site["name"]
        self.short_name = site["shortname"]
        self.vod_path = site["vod_path"]
        self.vod_baseline_path = site["baseline_path"]
        self.vod_timeseries_path = site["timeseries_path"]
        self.vod_product_path = site["product_path"]
        self.baseline_days = site["baseline_days"]
        self.paired_path = site["paired_path"]
        self.plot_path = site["plot_path"]
        self.date_next = site["date_next"]
        self.splitter_raw = site["splitter_raw"].split("&")

        self.grnd_inport_pattern = site["grnd_inport_pattern"]
        self.grnd_raw_path = site["grnd_raw_path"]
        self.grnd_temp_path = site["grnd_temp_path"]

        self.twr_inport_pattern = site["twr_inport_pattern"]
        self.twr_raw_path = site["twr_raw_path"]
        self.twr_temp_path = site["twr_temp_path"]

        # Handle variables
        self.keep_vars = site["keep_vars"].split("-")


    INTERVAL = "60s"

    def preprocess(self, is_tower, start_date, is_autotime):
        start_time = datetime.datetime.now()

        if is_tower:
            location = "Twr"
            temppath = self.twr_temp_path
            rawpath = self.twr_raw_path
            name = self.twr_inport_pattern
        else:
            location = "Grnd"
            temppath = self.grnd_temp_path
            rawpath = self.grnd_raw_path
            name = self.grnd_inport_pattern

        if is_autotime:
            start_date = pd.Timestamp(f'{self.date_next} 00:00:00')

        time_period = pd.interval_range(start=start_date, periods=7, freq='D')

        input_pattern = {name: rawpath}
        outdir = {name: temppath}

        logging.info(f"Preprocess of site {self.name} - {location} to {outdir} between {time_period[0].left} and {time_period[-1].right}")
        print(f"Preprocess of site {self.name} - {location} to {outdir}")
        preprocess(input_pattern, True, interval=self.INTERVAL, outputdir=outdir,
                   keepvars=self.keep_vars, unzip_path=temppath, time_period=time_period, splitter=self.splitter_raw)

        extension = '.rnx'
        self.delete_files(temppath, extension)
        if is_autotime:
            self.adjust_autodate(time_period[-1].right)

        end_time = datetime.datetime.now()
        elapsed_time = end_time - start_time
        print(f"Preprocessing finished")
        print(f"Time used: {elapsed_time} seconds")


    def pairing(self, time_period=None):
        extension = self.get_extension("nc")

        pairings = {self.short_name: (self.twr_inport_pattern, self.grnd_inport_pattern)}
        filepattern = {self.twr_inport_pattern: self.twr_temp_path+extension,
                       self.grnd_inport_pattern: self.grnd_temp_path+extension}
        outputdir = {self.short_name: self.paired_path}

        keep_vars = self.keep_vars
        # keep_vars = []
        # for pair_var in pair_vars:
        #     keep_vars.append(pair_var+"_ref")
        #     keep_vars.append(pair_var + "_grn")

        logging.info(f"Pairing of site {self.name} - {pairings} between {time_period[0].left} and {time_period[-1].right}")
        print(f"Pairing of site {self.name}")
        print(f"{pairings} between {time_period[0].left} and {time_period[-1].right}")
        gather_stations(filepattern, pairings, time_period, outputdir=outputdir, splitter=self.splitter_raw) #, time_period=time_period)


    def calculate_vod(self, time_periode=None):
        extension = self.get_extension("nc")
        filepattern = {self.short_name: self.paired_path+extension}
        outputdir = {self.short_name: self.vod_path}
        pairing = {self.short_name: (self.twr_inport_pattern, self.grnd_inport_pattern)}
        bands = {self.short_name: self.keep_vars}

        logging.info(f"Calculate VOD of site {self.name} between {time_periode[0].left} and {time_periode[-1].right}")
        print(f"Calculate VOD: {filepattern}")
        process_vod(filepattern, pairing, bands, time_periode, outputdir)

    def get_extension(self, ext):
        extension = f"/*.{ext}" # Linux extension
        if os.name == 'nt':  # 'nt' represents Windows
            extension = f"\\*.{ext}"
        return extension

    def delete_files(self, directory, extension):
        # List all files in the directory
        files = os.listdir(directory)
        print(f"Delete all {extension} files at {directory}")

        # Iterate through the files and delete those with the specified extension
        for file in files:
            if file.endswith(extension):
                file_path = os.path.join(directory, file)
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")


    def adjust_autodate(self, new_date):
        def overwrite_config_line(config_path, section, option, new_value):
            # Create a ConfigParser object
            config = configparser.ConfigParser()

            # Read the existing configuration file
            config.read(config_path)

            # Update the value in the specified section and option
            config.set(section, option, new_value)

            # Write the updated configuration back to the file
            with open(config_path, 'w') as config_file:
                config.write(config_file)

        config_path = 'config.ini'
        section = self.config_parser.name
        option = 'date_next'
        new_value = new_date.strftime("%Y-%m-%d")

        overwrite_config_line(config_path, section, option, new_value)
        

    def create_timeseries(self, year):
        print(f"Create VOD times series")
        in_path = {self.short_name: self.vod_path}
        out_baseline_path = {self.short_name: self.vod_baseline_path}
        out_timeseries_path = {self.short_name: self.vod_timeseries_path}

        result = vod_timeseries.calc_timeseries(in_path, year, int(self.baseline_days), out_baseline_path, out_timeseries_path)


    def create_product(self, hour_frequency=1):
        print(f"Create VOD product")
        in_path = {self.short_name: self.vod_timeseries_path}
        out_path = {self.short_name: self.vod_product_path}

        do_rain = True
        if do_rain:
            t=1
            # prc = pd.read_csv(
            #     r"S:\group\rsws\Data\Sites\CH-LAE_Laegeren\ETH_tower_measurements\NewPlatform_2021-2024\precip_cummulative_mm.csv")
            # prc["Epoch"] = pd.to_datetime(prc["Time"], format='%Y-%m-%d %H:%M:%S')
            # prc = prc[["Epoch", "CH-LAE"]]
            # prc = prc[prc["CH-LAE"] > 0]
            # prc['prec'] = prc['CH-LAE'].fillna(method='ffill').diff()
            # prc.loc[prc['prec'].isnull(), 'prec'] = prc['CH-LAE']
            # prc = prc[["Epoch", "prec"]]
            # prc_hourly = prc.groupby(pd.Grouper(freq='1h', key='Epoch')).sum()
            # prc_hourly.to_csv(r'S:\group\rsws_gnss\Meteo_data\Laeg\Laeg_precip.csv', index=True)

        result = vod_timeseries.calc_product(in_path, self.baseline_days, hour_frequency, out_path)



    def plot_timeseries(self, year, time_periode=None):
        print(f"Plot VOD times series")
        timeseries_path = {self.short_name: self.vod_timeseries_path}
        product_path = {self.short_name: self.vod_product_path}
        baseline_path = {self.short_name: self.vod_baseline_path}

        out_path = {self.short_name: self.plot_path}
        #t_path = r'X:\rsws_gnss\VOD_timeseries_live/Lae_VOD_timeseries_bl15days_2023.nc'
        result = vod_plots.do_plot(timeseries_path, product_path, baseline_path, year, self.baseline_days, out_path)


    def plot_analysis(self):
        print(f"Plot VOD times series")
        timeseries_path = {self.short_name: self.vod_timeseries_path}
        product_path = {self.short_name: self.vod_product_path}
        baseline_path = {self.short_name: self.vod_baseline_path}

        out_path = {self.short_name: self.plot_path}
        result = vod_plots.analysis_plot(product_path, self.baseline_days, out_path)
