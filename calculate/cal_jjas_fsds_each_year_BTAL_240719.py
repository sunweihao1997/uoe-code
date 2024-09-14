'''
2024-7-19
This script is to calculate the JJAS mean FSDS for BTAL experiment
'''
import xarray as xr
import numpy as np
from netCDF4 import Dataset

file0 = xr.open_dataset('/exports/csce/datastore/geos/users/s2618078/data/analysis_data/BTAL_FSDS_ensemble_mean_231005.nc')

# === Claim the array saving the result ===
#print(len(file0['time'].data)/12) #156 Years. Here I only calculate 150 years, and the begin is 1850-01
jjas_prect = np.zeros((157, 96, 144))

data_JJAS = file0.sel(time=file0.time.dt.month.isin([6, 7, 8, 9]))

# === Calculation ===
for i in range(157):
    #a = np.average(file0['PRECC'].data[first_mon : first_mon + 3], axis=0)
    #print(a.shape)
    jjas_prect[i, :, :]  =  np.average(data_JJAS['FSDS'].data[i*4 : i*4 + 4], axis=0)

# === Write to ncfile ===
ncfile  =  xr.Dataset(
{
    "FSDS_JJAS": (["time", "lat", "lon"], jjas_prect),
},
coords={
    "time": (["time"], np.linspace(1850, 1850+156, 157)),
    "lat":  (["lat"],  file0['lat'].data),
    "lon":  (["lon"],  file0['lon'].data),
},
)

ncfile['FSDS_JJAS'].attrs = file0['FSDS'].attrs


ncfile.attrs['description']  =  'Created on 2023-12-05. This file is the JJAS ensemble mean among the 8 member in the BTAL emission experiments. The variables is FSDS. This file is the modified type for the BTAL_FSDS_jja_mean_231005.nc'
ncfile.to_netcdf("/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/BTAL_FSDS_jjas_mean_231205.nc", format='NETCDF4')