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
import gnssvod.analysis.compare_sensors as compare_sensors


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

        self.switch = None

        # Handle variables
        if site["keep_vars"] == "None":
            self.keep_vars = None
        else:
            self.keep_vars = site["keep_vars"].split("-")


    INTERVAL = "15s"

    def preprocess(self, is_tower, start_date, is_autotime, adjust_autotime=False):
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
        if is_autotime and adjust_autotime:
            self.adjust_autodate(time_period[-1].right)

        end_time = datetime.datetime.now()
        elapsed_time = end_time - start_time
        print(f"Preprocessing finished")
        print(f"Time used: {elapsed_time} seconds")


    def pairing(self, time_period=None, compress=True):
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

        # FIX of mixing two devices and antenna cable
        if self.switch is not None:
            time_periods = self.split_interval_index_by_dates(time_period, self.switch["date_start"], self.switch["date_end"])
        else:
            time_periods = [(time_period, 'normal')]

        logging.info(f"Pairing of site {self.name} - {pairings} between {time_period[0].left} and {time_period[-1].right}")
        print(f"Pairing of site {self.name}")

        for t, mode in time_periods:
            fpat = filepattern
            pair = pairings
            split = self.splitter_raw
            out_station_names=None
            if mode == "othersite":
                out_station_names = [self.twr_inport_pattern, self.grnd_inport_pattern]
                if self.switch["sensor"] == "twr":
                    pair = {self.short_name: (self.switch["othersite"].twr_inport_pattern, self.grnd_inport_pattern)}
                    fpat = {self.switch["othersite"].twr_inport_pattern: self.switch["othersite"].twr_temp_path + extension,
                                   self.grnd_inport_pattern: self.grnd_temp_path + extension}
                    split = self.splitter_raw + self.switch["othersite"].splitter_raw
                else:
                    pair = {self.short_name:
                                (self.twr_inport_pattern, self.switch["othersite"].grnd_inport_pattern)}
                    fpat = {self.twr_inport_pattern: self.twr_temp_path + extension,
                                   self.switch["othersite"].grnd_inport_pattern: self.switch["othersite"].grnd_temp_path + extension}
                    split = self.splitter_raw + self.switch["othersite"].splitter_raw
            print(f"{pair} between {t[0].left} and {t[-1].right} with {fpat}")
            # Only this one needed without switch
            gather_stations(fpat, pair, t, compress=compress,
                            outputdir=outputdir, splitter=split, out_station_names=out_station_names)


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
        

    def create_timeseries(self, year, ignore_glosas=False):
        print(f"Create VOD times series")
        in_path = {self.short_name: self.vod_path}
        out_baseline_path = {self.short_name: self.vod_baseline_path}
        out_timeseries_path = {self.short_name: self.vod_timeseries_path}

        if ignore_glosas:
            print("Ignore Glossas satellites")

        result = vod_timeseries.calc_timeseries(in_path, year, int(self.baseline_days), out_baseline_path, out_timeseries_path, ignore_glosas=ignore_glosas)


    def create_product(self, hour_frequency=1, ignore_glosas=False, max_ele=90, only_const=None):
        print(f"Create VOD product")
        in_path = {self.short_name: self.vod_timeseries_path}
        out_path = {self.short_name: self.vod_product_path}

        print(f"Do time frequency: {hour_frequency}")
        if ignore_glosas:
            print("Ignore Glossas satellites")
        if max_ele < 90:
            print(f"Limit elevation to: {max_ele}")
        if only_const is not None:
            print(f"Only do constellation: {only_const}")
        print(f"Output to: {out_path}")

        do_rain = False
        if do_rain:
            print("do rain calc")
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

        result = vod_timeseries.calc_product(in_path, self.baseline_days, hour_frequency, out_path,
                                             ignore_glosas=ignore_glosas, max_elevation=max_ele, only_const=only_const)



    def plot_timeseries(self, year, ignore_glosas=False, max_ele=90):
        print(f"Plot VOD times series")
        timeseries_path = {self.short_name: self.vod_timeseries_path}
        product_path = {self.short_name: self.vod_product_path}
        baseline_path = {self.short_name: self.vod_baseline_path}

        out_path = {self.short_name: self.plot_path}
        result = vod_plots.do_plot(timeseries_path, product_path, baseline_path, year, self.baseline_days, out_path, ignore_glosas=ignore_glosas, max_elevation=max_ele)


    def plot_analysis(self, ignore_glosas=False, max_ele=90):
        print(f"Plot VOD times series")
        timeseries_path = {self.short_name: self.vod_timeseries_path}
        product_path = {self.short_name: self.vod_product_path}
        baseline_path = {self.short_name: self.vod_baseline_path}

        out_path = {self.short_name: self.plot_path}
        result = vod_plots.analysis_plot(product_path, self.baseline_days, out_path, ignore_glosas=ignore_glosas, max_elevation=max_ele)

    def compare(self):
        extension = self.get_extension("nc")
        filepattern = {self.short_name: self.paired_path+extension}
        pairing = {self.short_name: (self.twr_inport_pattern, self.grnd_inport_pattern)}
        bands = {self.short_name: self.keep_vars}

        compare_sensors.do_comparison(filepattern, pairing, bands)

    def plot_compare_vod(self, cosites_path, year, ignore_glosas=False, max_ele=90):
        print(f"Plot compare VOD times series")
        # product_path = {self.short_name: self.vod_product_path}
        # cosites_path[self.short_name] = self.vod_product_path
        compare_sensors.compare_vod_annual(cosites_path, year, self.baseline_days, self.plot_path, ignore_glosas=ignore_glosas, max_elevation=max_ele)

    def plot_satconstellation(self):
        print(f"Plot satellite constellation VOD times series")
        vod_plots.plot_constellations(self.short_name, self.vod_product_path, self.baseline_days)

    def plot_hemi_satconstellation(self):
        print(f"Plot satellite constellation VOD hemisphere ")
        vod_plots.plot_hemi_sat_constellation(self.short_name)

    def compare_sat_constellation(self):
        print(f"compare satellite constellation VOD ")
        filepattern = {self.short_name: self.vod_timeseries_path}
        compare_sensors.compare_sv_const(filepattern)

    def split_interval_index_by_dates(self,
        interval_index: pd.IntervalIndex,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp
    ) -> [(pd.IntervalIndex, str)]:
        """
        Splits a pandas IntervalIndex based on a start and end date,
        using an alternative method for older pandas versions.

        Args:
            interval_index: The pandas.IntervalIndex to split.
            start_date: The start date for the splitting logic.
            end_date: The end date for the splitting logic.

        Returns:
            A list of pandas.IntervalIndex objects.
        """
        # Initialize a list to hold the newly created intervals
        start_loc, end_loc = None, None
        start_in_index, end_in_index = False, False

        try:
            start_loc = interval_index.get_loc(start_date)
            start_in_index = True
        except KeyError:
            pass

        try:
            end_loc = interval_index.get_loc(end_date)
            end_in_index = True
        except KeyError:
            pass

        # --- Helper to create IntervalIndex for older pandas ---
        def create_interval_index(intervals):
            if not intervals:
                return pd.IntervalIndex.from_intervals([], closed='left')
            return pd.Index(intervals, dtype=interval_index.dtype)

        # --- NEW CASE: Dates encompass the entire index ---
        if not start_in_index and not end_in_index:
            if start_date < interval_index.left[0] and end_date > interval_index.right[-1]:
                return [(interval_index, 'othersite')]
            # --- ORIGINAL CASE 1: No dates fall within, but they don't encompass it either ---
            else:
                return [(interval_index, 'normal')]

        # --- CASE 2: One date falls within the index. Split into two parts, both 'outside'. ---
        elif start_in_index and not end_in_index:
            before_split_intervals = interval_index[:start_loc].tolist()
            interval_to_split = interval_index[start_loc]
            before_split_intervals.append(pd.Interval(interval_to_split.left, start_date))

            after_split_intervals = [pd.Interval(start_date, interval_to_split.right)]
            after_split_intervals.extend(interval_index[start_loc + 1:].tolist())

            return [
                (create_interval_index(before_split_intervals), 'normal'),
                (create_interval_index(after_split_intervals), 'othersite')
            ]

        elif not start_in_index and end_in_index:
            before_split_intervals = interval_index[:end_loc].tolist()
            interval_to_split = interval_index[end_loc]
            before_split_intervals.append(pd.Interval(interval_to_split.left, end_date))

            after_split_intervals = [pd.Interval(end_date, interval_to_split.right)]
            after_split_intervals.extend(interval_index[end_loc + 1:].tolist())

            return [
                (create_interval_index(before_split_intervals), 'othersite'),
                (create_interval_index(after_split_intervals), 'normal')
            ]

        # --- CASE 3: Both dates fall within the index. Split into three parts. ---
        elif start_in_index and end_in_index:
            part1_intervals = interval_index[:start_loc].tolist()
            interval_start = interval_index[start_loc]
            part1_intervals.append(pd.Interval(interval_start.left, start_date))

            part2_intervals = []
            if start_loc == end_loc:
                part2_intervals.append(pd.Interval(start_date, end_date))
            else:
                part2_intervals.append(pd.Interval(start_date, interval_start.right))
                part2_intervals.extend(interval_index[start_loc + 1:end_loc].tolist())

                interval_end = interval_index[end_loc]
                part2_intervals.append(pd.Interval(interval_end.left, end_date))

            part3_intervals = []
            if start_loc != end_loc:
                interval_end = interval_index[end_loc]
                part3_intervals.append(pd.Interval(end_date, interval_end.right))

            part3_intervals.extend(interval_index[end_loc + 1:].tolist())

            return [
                (create_interval_index(part1_intervals), 'normal'),
                (create_interval_index(part2_intervals), 'othersite'),
                (create_interval_index(part3_intervals), 'normal')
            ]

        # Case 4 all fall inside
        return [(interval_index, 'normal')]