#####################################################
# HELPER FUNCTION FOR THE VOD TIME SERIES CALCULATION
#####################################################
# Multiple functions that are needed load and process pre-processed VOD monthly data files to a timeseries.
import os.path

# --------------------
# SET UP THE SCRIPT
# --------------------
import gnssvod.hemistats.hemistats as hemistats
import glob
import re
import pandas as pd
from datetime import datetime, timedelta
import xarray as xr

# --------------------
# Index available data
# --------------------
def index_data_files(in_path, station_name, mode="date"):
    """
    Stores the filepath of all available .nc files in a folder and extracts the start and end date from the string.

    Parameters
    ----------
    in_path: folder path as string

    Returns
    -------
    file_dates: dataframe
        The start and end dates of the files are stored as string as well as a Timestamp

    """
    # List the file names of all VOD files in the folder path
    files = glob.glob(in_path + r'\*.nc')

    # Get all files
    if mode == "all":
        return files

    # Get annual files
    if mode == "annual":
        years = []
        for file in files:
            year = file.split(f"{station_name}_")[1][:4]
            years.append(year)
        file_dates = pd.DataFrame({'File': files, 'Year': years})
        return file_dates

    # Get monthly files
    start = []
    end = []
    # Extract the start and end date from the file name and store in list
    for file in files:
        start.append(re.search(rf'({station_name}_)(\d+)_', file).group(2))
        end.append(re.search(r'_(\d+).nc', file).group(1))

    file_dates = pd.DataFrame({'File': files, 'Start': start, 'End': end})
    # Add the start and end as timestamp
    file_dates['StartDT'] = pd.to_datetime(file_dates['Start'], format='%Y%m%d')
    file_dates['EndDT'] = pd.to_datetime(file_dates['End'], format='%Y%m%d')
    # Add timedelta of one day minus one second to set the time to 23:59:59
    file_dates['EndDT'] = file_dates['EndDT'] + timedelta(days=1) - timedelta(seconds=1)

    return file_dates


# --------------------
# Define time period
# --------------------
def timePeriod(start, end):
    """
    Generate a timeperiod index to loop through based on the start and end date selected

    Parameters
    __________
    start, end: string
        Start and end dates formatted as 'yyyy-mm-dd'

    Returns
    -------
    timeperiod: interval[datetime64[ns], right]
        IntervalIndex with each day in the selected time period from 00:00:00 to 23:59:59

    """

    # Calculate the number of days between start and end (inclusive)
    num_days = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1

    # Create an empty list to store intervals
    timeperiod = []

    # Loop through each day from start to end (inclusive)
    for i in range(num_days):
        # Calculate the current day's date
        current_date = pd.Timestamp(start) + timedelta(days=i)

        # Create a daily interval with timestamps from 00:00:00 to 23:59:59
        day_interval = pd.Interval(left=current_date,
                                   right=current_date + timedelta(seconds=86399))  # 86399 seconds = 23:59:59
        timeperiod.append(day_interval)

    # Combine the intervals into a DatetimeIndex
    timeperiod = pd.IntervalIndex(timeperiod)

    return timeperiod


# --------------------
# Open VOD files
# --------------------
def open_data(filepath):
    """
    Opens VOD netCDF files, converts to a pandas data frame and drops the not needed data.
    """
    # Load the processed VOD data set
    ds = xr.open_dataset(filepath)

    # Convert the xarray to a pandas data frame, sorted by Epoch and satellite (Coordinates of ds). All Data variables of
    # ds_new are now columns in the data frame.
    df = ds.to_dataframe().dropna(how='all').reorder_levels(['Epoch', 'SV']).sort_index()

    # Subset the columns that we need
    df_new = df[['VOD', 'Azimuth', 'Elevation']]

    return df_new


# --------------------
# Calculate the baseline VOD
# --------------------

def VOD_base_calc(file_dates, timeperiod, bl_kernel, out_path, bl_name, save_bs=False):
    """
    Calculates a baseline VOD in the desired kernel size for each day in the selected time period.

    Parameters
    ----------
    file_dates: Data frame
        Lists the data paths of the VOD files and their start and end times. Return of the index_data_files funct.

    timeperiod: interval[datetime64[ns], right]
        IntervalIndex with each day in the selected time period from 00:00:00 to 23:59:59. Return of the timePeriod funct.

    bl_kernel: int
        Number of days to both sides of the selected day to use as the baseline. E.g. bl_kernel = 7 gives a baseline of
        15 days (7 left, day itself, 7 right).

    out_path: string
        Path to store the data

    bl_name: string
        Name of the output file

    save_bs: Boolean
        If selected the generated multi-index data frame will be stored as netCDF in the defined folder.

    Returns
    -------
    VOD_baseline: multi-index data frame
        The baseline values are stored per hemisphere cell (CellID) and day in the time period (Day) selected.

    """
    print(f"Calculate Kernel {bl_kernel} days baseline of {timeperiod.left[0]} - {timeperiod.right[-1]}")
    # List to store VOD baseline files (per day)
    daily_data = []

    # Track the number of skipped days with no baseline
    skipped_days = 0

    # Loop through the time period
    for day in timeperiod:
        # Define baseline and calculate buffered interval-> the bl_kernel will be added on both sides
        baseline = timedelta(days=bl_kernel)
        baseline_starday = day.left - baseline
        baseline_endday = day.right + baseline

        # Check if 'day' +/- baseline contains any datetime in StartDT or EndDT in the generated file_dates df
        is_between = ((file_dates['StartDT'] <= baseline_starday) & (file_dates['EndDT'] >= baseline_starday)) | \
                     ((file_dates['StartDT'] <= day.left) & (file_dates['EndDT'] >= day.right)) | \
                     ((file_dates['StartDT'] <= baseline_endday) & (file_dates['EndDT'] >= baseline_endday))

        # Check if the day is between the StartDT and EndDT of any file
        matching_row = file_dates[is_between]  # Get the matching row(s)

        if not matching_row.empty:
            print(f'Opening {len(matching_row)} file(s) for baseline around day {day.left}')

            # Generate a list to store the opened files
            data_list = []

            # Open the file paths in the matching rows
            for i in range(0, len(matching_row)):
                # print(f'Opening file(s): {matching_row["File"].iloc[i]}.')
                file = open_data(matching_row['File'].iloc[i])
                data_list.append(file)

            # Concat the two dataframes together
            df = pd.concat(data_list)

            # Select a subset of df based on the baseline
            subset_df = df.loc[(slice(baseline_starday, baseline_endday),), :]

            # Check if the subset has values (here one could also set another threshold for min. days in baseline)
            if not subset_df.empty:

                # Calculate the VOD average per cell in hemisphere based on the baseline interval
                print('Calculating the baseline VOD.')

                # Initialize hemispheric grid
                hemi = hemistats.hemibuild(2)

                # Classify vod data into grid cells and drop the azm und ele columns after
                vod = hemi.add_CellID(subset_df, aziname='Azimuth', elename='Elevation').drop(
                    columns=['Azimuth', 'Elevation'])

                # Get mean, std and count values per grid cell
                vod_avg = vod.groupby(['CellID']).agg(['mean', 'std', 'count'])

                # Flatten the columns
                vod_avg.columns = ["_".join(x) for x in vod_avg.columns.to_flat_index()]

                # Generate a date object based on 'day'
                start_ts = day.left
                date_obj = start_ts.date()

                # Add the date object to a DataFrame column
                vod_avg['Day'] = date_obj

                # Convert to DateTime
                vod_avg['Day'] = pd.to_datetime(vod_avg['Day'])

                # Creating multiindex data frame
                new_index = pd.MultiIndex.from_tuples([(date, cell_id) for cell_id, date in zip(vod_avg.index, vod_avg['Day'])])
                vod_avg = vod_avg.set_index(new_index)
                vod_avg.index = vod_avg.index.set_names(['Date', 'CellID'])

                # Add the generated baseline file to the list
                daily_data.append(vod_avg)

            else:
                print(f'Not enough data for baseline on day {day.left}. Skipping...')
                skipped_days += 1

        else:
            print(f'No matching file(s) for day: {day.left}')

    print(f'\nTotal skipped days without baseline: {skipped_days}')

    # Combine VOD baseline dfs
    VOD_baseline = pd.concat(daily_data, axis=0)

    # Drop the 'Day' column again?
    VOD_baseline = VOD_baseline.drop(columns=['Day'])

    if save_bs:
        # Store the VOD_baseline file as a .nc
        ds = xr.Dataset.from_dataframe(VOD_baseline) # change to xarray
        comp = dict(zlib=True, complevel=5)
        encoding = {var: comp for var in ds.data_vars}

        # Save as NetCDF file
        filepath = os.path.join(out_path, bl_name + '.nc')
        print(f'Writing the baseline VOD file to {filepath}')
        ds.to_netcdf(filepath, format="NETCDF4", engine="netcdf4", encoding=encoding)

    return VOD_baseline

# --------------------
# Calculate time series
# --------------------
def vod_timeseries_baseline_correction(file_dates, timeperiod, bl_data, out_path, ts_name, save_ts=False):
    """
        Calculates the VOD timeseries in the selected time period with the selected baseline file. Uncorrected and
        baseline corrected VOD means have the same length (same time period).

        Parameters
        ----------
        file_dates: Data frame
            Lists the data paths of the VOD files and their start and end times. Return of the index_data_files funct.

        timeperiod: interval[datetime64[ns], right]
            IntervalIndex with each day in the selected time period from 00:00:00 to 23:59:59. Return of the timePeriod funct.

        bl_data: DataFrame
            Baseline data calculated before with VOD_base_calc

        out_path: string
            Path to store the data

        ts_name: string
            Name of the output file

        save_ts: Boolean
            If selected the generated multi-index data frame will be stored as netCDF in the defined folder.

        Returns
        -------
        VOD_anom: multi-index data frame
            The daily VOD values are combined with the respective baseline values and an anomaly value is calculated.

        """
    print(f"Calculate VOD time series of {timeperiod}")

    # Generate a list to store the daily anomaly dfs
    daily_vod_anom = []

    # Track the number of skipped days with no data
    skipped_days = 0

    # Loop through the days in the selected time period
    for day in timeperiod:
        # Find the correct file to open based on the current day in the timeperiod
        # Check if 'day' contains any datetime in StartDT or EndDT in the generated file_dates df
        is_between = (file_dates['StartDT'] <= day.left) & (file_dates['EndDT'] >= day.right)

        # Check if the day is between the StartDT and EndDT of any file
        matching_row = file_dates[is_between]  # Get the matching row(s)

        if not matching_row.empty:
            print(f'Opening {len(matching_row)} file(s) for baseline around day {day.left}')

            # Open the matching files
            data_list = []
            for i in range(0, len(matching_row)):
                print(f'Opening file(s): {matching_row["File"].iloc[i]}.')
                file = open_data(matching_row['File'].iloc[i])
                data_list.append(file)
            df = pd.concat(data_list)

            # Subset the daily and the baseline VOD file to the date (if the day has measurements)
            date_string = day.left.strftime('%Y-%m-%d')

            # Access data for existing days and skip the ones that are not present
            try:
                vod_day = df.xs(date_string, level='Epoch')
                vod_bl = bl_data.xs(day.left, level='Date')

                # Initialize hemispheric grid
                hemi = hemistats.hemibuild(2)

                # Classify vod data into grid cells and drop the azm und ele columns after
                vod = hemi.add_CellID(vod_day, aziname='Azimuth', elename='Elevation').drop(
                    columns=['Azimuth', 'Elevation'])

                # Merge statistics with the original VOD measurements
                vod_anom_day = vod.join(vod_bl, on='CellID')

                # Add to the daily file to the list
                daily_vod_anom.append(vod_anom_day)

            except KeyError:
                print(f'No data found for {date_string}. Skipping...')
                skipped_days += 1

    print(f'\nTotal skipped days: {skipped_days}')

    # Combine the daily vod_anom fields
    VOD_anom = pd.concat(daily_vod_anom, axis=0)

    # Calculate the anomaly
    VOD_anom['VOD_anom'] = VOD_anom['VOD'] - VOD_anom['VOD_mean']

    if save_ts:
        # Store the VOD_baseline file as a .nc
        ds = xr.Dataset.from_dataframe(VOD_anom) # change to xarray
        comp = dict(zlib=True, complevel=5)
        encoding = {var: comp for var in ds.data_vars}

        # Save as NetCDF file
        filepath = os.path.join(out_path, ts_name + '.nc')
        print(f'Writing the VOD timeseries file to {filepath}')
        ds.to_netcdf(filepath, format="NETCDF4", engine="netcdf4", encoding=encoding)

    return VOD_anom








