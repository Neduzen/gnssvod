import configparser
import datetime
import logging
import os
import sys
import datetime as datetime
from gnssvod.io.preprocess import (preprocess, get_filelist, pair_obs, calc_vod)
import pandas as pd

class Gnss_site:
    def __init__(self, site):
        self.config_parser = site
        self.name = site["name"]
        self.short_name = site["shortname"]
        self.vod_path = site["vod_path"]
        self.paired_path = site["paired_path"]
        self.date_next = site["date_next"]

        self.grnd_inport_pattern = site["grnd_inport_pattern"]
        self.grnd_raw_path = site["grnd_raw_path"]
        self.grnd_temp_path = site["grnd_temp_path"]

        self.twr_inport_pattern = site["twr_inport_pattern"]
        self.twr_raw_path = site["twr_raw_path"]
        self.twr_temp_path = site["twr_temp_path"]

        # Keep vars
        self.keep_vars  = site["keep_vars"].split("-")
        self.pairing_vars = site["pairing_vars"].split("-")


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
                   keepvars=self.keep_vars, unzip_path=temppath, time_period=time_period)

        extension = '.rnx'
        self.delete_files(temppath, extension)
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

        logging.info(f"Pairing of site {self.name} - {pairings} between {time_period[0].left} and {time_period[-1].right}")
        print("Pairing: " + str(pairings))
        pair_obs(filepattern, pairings, time_period, outputdir=outputdir, time_period=time_period)


    def calculate_vod(self, time_periode=None):
        extension = self.get_extension("nc")
        filepattern = {self.short_name: self.paired_path+extension}
        pairing = {self.short_name: ('S1_ref', 'S1_grn', 'Elevation_grn')}
        outputdir = {self.short_name: self.vod_path}

        logging.info(f"Calculate VOD of site {self.name} between {time_periode[0].left} and {time_periode[-1].right}")
        print(f"Calculate VOD: {filepattern}")
        result = calc_vod(filepattern, pairing, outputdir, time_periode)
    
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