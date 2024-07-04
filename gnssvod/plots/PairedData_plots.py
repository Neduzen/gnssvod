## CREATE HEMISPHERIC PLOTS FROM PAIRED DATA ##
# Use the code from the example scripts provided by Vincent and adjusted to our processing

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

# ----------------
# Paired data set
# ----------------
# Load all paired data sets in the folder as series of xarrays based on the time variable 'Epoch'
ds = xr.open_mfdataset(r'C:/Users/jkesselr/Documents/PhD/Data/Objective2/GNSS/Laegern/Laeg_*.nc', combine='nested', concat_dim='Epoch')

# We don't have the station object. All variables are still _ref (tower) and _grn. We need to rearrange the xarray and
# add the additional level.
# Concatenate S1a and S1b along a new dimension 'station'
s1_combined = xr.concat([ds['S1_ref'], ds['S1_grn']], dim='Station')
s1_combined['Station'] = ['twr', 'grn']
s2_combined = xr.concat([ds['S2_ref'], ds['S2_grn']], dim='Station')
s2_combined['Station'] = ['twr', 'grn']
sys_combined = xr.concat([ds['SYSTEM_ref'], ds['SYSTEM_grn']], dim='Station')
sys_combined['Station'] = ['twr', 'grn']
ele_combined = xr.concat([ds['Elevation_ref'], ds['Elevation_grn']], dim='Station')
ele_combined['Station'] = ['twr', 'grn']
azi_combined = xr.concat([ds['Azimuth_ref'], ds['Azimuth_grn']], dim='Station')
azi_combined['Station'] = ['twr', 'grn']
# Update dimension coordinates
S1 = xr.DataArray(s1_combined, dims=('Station', 'Epoch', 'SV'), name='S1')
S2 = xr.DataArray(s2_combined, dims=('Station', 'Epoch', 'SV'), name='S2')
ele = xr.DataArray(ele_combined, dims=('Station', 'Epoch', 'SV'), name='Elevation')
azi = xr.DataArray(azi_combined, dims=('Station', 'Epoch', 'SV'), name='Azimuth')
sys = xr.DataArray(sys_combined, dims=('Station', 'Epoch', 'SV'), name='SYSTEM')
# Combine DataArrays into one Dataset
ds_new = xr.merge([S1, S2, ele, azi, sys])

# Convert the xarray to a pandas data frame, sorted by Epoch and satellite (Coordinates of ds). All Data variables of
# ds_new are now columns in the data frame.
df = ds_new.to_dataframe().dropna(how='all').reorder_levels(['Station', 'Epoch', 'SV']).sort_index()

# ----------------------------------------------------------------------------------------------
# Make a figure of the subset
mySV = 'G03'
mystation_name = 'grn'
day = '2023-07-05'

# subset the dataset for plotting e.g. Galileo #3 ground station 1 day
subdf = df.xs(mystation_name,level='Station').xs(day, level='Epoch').xs(mySV,level='SV')

# Initialize figure with polar axes
fig, ax = plt.subplots(figsize=(10,8),subplot_kw=dict(projection='polar'))

# Polar plots need a radius and theta direction in radians
radius = 90-subdf.Elevation
theta = np.deg2rad(subdf.Azimuth)

# Plot each measurement and color by signal to noise ratio
hs = ax.scatter(theta,radius,c=subdf.S1)
ax.set_rlim([0,90])
ax.set_theta_zero_location("N")
plt.colorbar(hs, shrink=.5, label='SNR (L1) [dB]')
#plt.title('Davos ground - ' + day + ': satellite ' + mySV, fontsize = 20)
plt.title('Laegern ground - ' + day + ': satellite ' + mySV, fontsize = 20)
plt.show()
#fig.savefig('DAVpaired_hemplot_1sat1day.png', bbox_inches = 'tight', pad_inches = 0.1, dpi = 300)
fig.savefig('Figures/LAEpaired_hemplot_1sat1day.png', bbox_inches = 'tight', pad_inches = 0.1, dpi = 300)

# ----------------------------------------------------------------------------------------------
# Time series of one satellite
mySV = 'G03'
day = '2023-07-05'

# Get all sites as list
station_names = df.index.get_level_values('Station').unique()

fig, ax = plt.subplots(1, figsize=(10, 5))
for i, iname in enumerate(station_names):
    # subset the dataset
    subdf = df.xs(day, level='Epoch').xs(iname, level='Station').xs(mySV, level='SV')
    # plot each measurement and color by signal to noise ratio
    hs = ax.plot(subdf.index.get_level_values('Epoch'), subdf.S1, label=iname)

myFmt = mdates.DateFormatter('%H:%M')
ax.xaxis.set_major_formatter(myFmt)
ax.set_ylabel('SNR (L1) [dB]', fontsize = 16)
ax.set_xlabel(day, fontsize = 16)
ax.legend()
plt.title('Satellite: ' + mySV, fontsize = 20)
plt.show()

# ----------------------------------------------------------------------------------------------
# Make a figure of all satellite observations
# Get all sites as list
station_names = df.index.get_level_values('Station').unique()
day = '2023-07-05'

# Ensure we use the same color limits in all plots
clim = [15,47]

# Initialize figure with polar axes
fig, ax = plt.subplots(1,len(station_names),figsize=(10,6),subplot_kw=dict(projection='polar'))
for i, iname in enumerate(station_names):
    # subset the dataset
    subdf = df.xs(iname,level='Station').xs(day, level='Epoch')
    # polar plots need a radius and theta direction in radians
    radius = 90-subdf.Elevation
    theta = np.deg2rad(subdf.Azimuth)
    # plot each measurement and color by signal to noise ratio
    hs = ax[i].scatter(theta,radius,c=subdf.S1,s=10)
    hs.set_clim(clim)
    ax[i].set_rlim([0,90])
    ax[i].set_theta_zero_location("N")
    ax[i].set_title(iname, fontsize = 16)

plt.colorbar(hs, ax=ax, location='bottom', shrink=.5, pad=0.05, label='SNR (L1) [dB]')
#plt.suptitle('Davos - ' + day, fontsize = 20)
plt.suptitle('Laegern - ' + day, fontsize = 20)
plt.show()
#fig.savefig('DAVpaired_hemplot_allSats1day.png', bbox_inches = 'tight', pad_inches = 0.1, dpi = 300)
fig.savefig('Figures/LAEpaired_hemplot_allSats1day.png', bbox_inches = 'tight', pad_inches = 0.1, dpi = 300)

# ----------------------------------------------------------------------------------------------
# Calculating and plotting hemispherical averages

# Drop the 'SYSTEM' column from the dataframe
df = df.drop('SYSTEM', axis=1)

# Make a hemispheric object
hemi = gv.hemibuild(4)

# Get patches
patches = hemi.patches()

# Assign the df values from e.g. 1 day to the hemispheric object
newdf = hemi.add_CellID(df.xs('2023-07-05', level='Epoch'))

# Calculate the average per grid cell
hemi_average = newdf.groupby(['CellID','Station']).mean()

# Make a figure of the average value per grid cell
fig, ax = plt.subplots(1,2,figsize=(10,6.5),subplot_kw=dict(projection='polar'))

station_names = df.index.get_level_values('Station').unique()
for i, iname in enumerate(station_names):
    # associate the mean values to the patches, join inner will drop patches with no data, making plotting slightly faster
    ipatches = pd.concat([patches,hemi_average.xs(iname, level='Station')],join='inner',axis=1)
    # plotting with colored patches
    pc = PatchCollection(ipatches.Patches,array=ipatches.S1,edgecolor='face', linewidth=1)
    pc.set_clim([25,50])
    ax[i].add_collection(pc)
    ax[i].set_rlim([0,90])
    ax[i].set_theta_zero_location("N")
    ax[i].set_title(iname, fontsize = 16)

plt.colorbar(pc, ax=ax, location='bottom', shrink=.5, pad=0.05, label='SNR (L1) [dB]')
#plt.suptitle('Davos - ' + day + ': Average SNR per cell', fontsize = 20)
plt.suptitle('Laegern - ' + day + ': Average SNR per cell', fontsize = 20)
plt.show()
#fig.savefig('DAVpaired_hemplot_averageSNR.png', bbox_inches = 'tight', pad_inches = 0.1, dpi = 300)
fig.savefig('Figures/LAEpaired_hemplot_averageSNR.png', bbox_inches = 'tight', pad_inches = 0.1, dpi = 300)

# Make a figure of the number of observations per grid cell
hemi_count = newdf.groupby(['CellID','Station']).count()

fig, ax = plt.subplots(1,2,figsize=(10,6.5),subplot_kw=dict(projection='polar'))

station_names = df.index.get_level_values('Station').unique()
for i, iname in enumerate(station_names):
    # associate the mean values to the patches, join inner will drop patches with no data, making plotting slightly faster
    ipatches = pd.concat([patches,hemi_count.xs(iname, level='Station')],join='inner',axis=1)
    # plotting with colored patches
    pc = PatchCollection(ipatches.Patches,array=ipatches.S1,edgecolor='face', linewidth=1)
    pc.set_clim([0,200])
    ax[i].add_collection(pc)
    ax[i].set_rlim([0,90])
    ax[i].set_theta_zero_location("N")
    ax[i].set_title(iname, fontsize = 16)

plt.colorbar(pc, ax=ax, location='bottom', shrink=.5, pad=0.05, label='# of obs')
#plt.suptitle('Davos - ' + day + ': No. of observations', fontsize = 20)
plt.suptitle('Laegern - ' + day + ': No. of observations', fontsize = 20)
plt.show()
#fig.savefig('DAVpaired_hemplot_NoObservations.png', bbox_inches = 'tight', pad_inches = 0.1, dpi = 300)
fig.savefig('Figures/LAEpaired_hemplot_NoObservations.png', bbox_inches = 'tight', pad_inches = 0.1, dpi = 300)