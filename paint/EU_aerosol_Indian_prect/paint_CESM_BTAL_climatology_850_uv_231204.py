'''
2023-12-4
This script is to paint the climatology 200 uv field in order to grasp the mean-state of the JJAS circulation at high level
'''
import xarray as xr
import numpy as np
import os
from matplotlib import cm
from matplotlib.colors import ListedColormap
import sys
from matplotlib import projections
import cartopy.crs as ccrs
import cartopy
from cartopy.util import add_cyclic_point
import matplotlib.pyplot as plt
sys.path.append('/exports/csce/datastore/geos/users/s2618078/uoe-code/module/')
from module_sun import set_cartopy_tick
from module_sun import check_path, add_vector_legend

# ================================ File location =========================================

path_src = '/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/'
file_src = 'EUI_CESM_BTAL_BTALnEU_850_level_but_single_level_ensemble_mean_UVWZT_and_ttest_1945_1960.nc'

# ========================================================================================

file0  =  xr.open_dataset(path_src + file_src)
lat    =  file0.lat.data
lon    =  file0.lon.data

# ================================ Painting ==============================================

def paint_jjas_diff(u, v,):
    '''This function paint the Diff aerosol JJA'''
    proj       =  ccrs.PlateCarree()
    ax         =  plt.subplot(projection=proj)

    # Create the subplot
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

    # Tick setting
    # extent
    lonmin,lonmax,latmin,latmax  =  0,150,-10,80
    extent     =  [lonmin,lonmax,latmin,latmax]

    set_cartopy_tick(ax=ax,extent=extent,xticks=np.linspace(0,140,8,dtype=int),yticks=np.linspace(-10,70,9,dtype=int),nx=1,ny=1,labelsize=10.5)

    # contourf for the geopotential height
    #im1  =  ax.contourf(lon, lat, v, levels=np.linspace(-1, 1, 11), cmap='coolwarm', alpha=1, extend='both')

    # stippling
    #plt.rcParams.update({'hatch.color': 'gray'})
    #sp  =  ax.contourf(lon, lat, p, levels=[0.1, 1], colors='none', hatches=['..'])

    # contour for the meridional wind
    #im2  =  ax.contour(lon, lat, z, 6, colors='green')
    #ax.clabel(im2, inline=True, fontsize=10)

    ax.coastlines(resolution='50m', lw=1.2)

    # Vector Map
    q  =  ax.quiver(lon, lat, u, v, 
        regrid_shape=20, angles='uv',        # regrid_shape这个参数越小，是两门就越稀疏
        scale_units='xy', scale=1.5,        # scale是参考矢量，所以取得越大画出来的箭头就越短
        units='xy', width=0.5,              # width控制粗细
        transform=proj,
        color='k', headlength = 5, headaxislength = 4, headwidth = 4, alpha=0.8)

    add_vector_legend(ax=ax, q=q, speed=1)

    ax.set_ylabel("BTAL Long-term Changes", fontsize=11)

    ax.set_title('Climatology', fontsize=12.5)
    ax.set_title('850 hPa', loc='right', fontsize=12.5)

    # Add colorbar
    #plt.colorbar(im1, orientation='horizontal')

    #plt.savefig('/exports/csce/datastore/geos/users/s2618078/paint/analysis_EU_aerosol_climate_effect/circulation/BTAL_climatology_JJAS_200_uv_circulation_JJAS.pdf')
    plt.savefig('test1.png', dpi=600)

ncfile  =  xr.open_dataset('/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/EUI_CESM_BTAL_BTALnEU_850_level_but_single_level_ensemble_mean_UVWZT_and_ttest_1945_1960.nc')

u_mean_BTAL = np.average(ncfile['btal_u'].data, axis=0)
v_mean_BTAL = np.average(ncfile['btal_v'].data, axis=0)

paint_jjas_diff(u_mean_BTAL, v_mean_BTAL)