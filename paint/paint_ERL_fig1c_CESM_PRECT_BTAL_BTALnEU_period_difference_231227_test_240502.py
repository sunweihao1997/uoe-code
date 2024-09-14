'''
2023-12-21
This script is accompanied with script paint_GPCC_PRECT_period_difference_1940_1960_231221.py on Huaibei
to compare the model simulationwith the observation-based data
'''
import xarray as xr
import numpy as np
from matplotlib import projections
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches
from scipy import stats

sys.path.append("/exports/csce/datastore/geos/users/s2618078/uoe-code/module/")
from module_sun import *

lonmin,lonmax,latmin,latmax  =  60,120,0,40
extent     =  [lonmin,lonmax,latmin,latmax]

levels = np.array([-1.8, -1.5, -1.2, -0.9, -0.6, -0.5, -0.3, -0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.6, 0.9, 1.2, 1.5, 1.8])
#levels = np.linspace(-1., 1., 21)

# ===================== Calculation for JJA and JJA precipitation period difference ===========================

data_path = "/exports/csce/datastore/geos/users/s2618078/data/analysis_data/"
data_file = "CESM_PRECT_BTAL_BTALnEU_JJA_JJAS_1850_2006.nc"

data = xr.open_dataset(data_path + data_file)
lat  = data.lat.data
lon  = data.lon.data

#print(data)
data_p1 = data.sel(time=slice(1905, 1920))
data_p2 = data.sel(time=slice(1945, 1965))
data_p3 = data.sel(time=slice(1945, 1960))

prect_BTAL_JJA_DIFF     = np.average(data_p2['PRECT_JJA_BTAL'].data, axis=0) - np.average(data_p1['PRECT_JJA_BTAL'].data, axis=0)
prect_BTALnEU_JJA_DIFF  = np.average(data_p2['PRECT_JJA_BTALnEU'].data, axis=0) - np.average(data_p1['PRECT_JJA_BTALnEU'].data, axis=0)
#prect_BTAL_JJA_DIFF    = np.average(data_p2['PRECT_JJA_BTAL'].data, axis=0) - np.average(data_p1['PRECT_JJA_BTAL'].data, axis=0)
#prect_BTALnEU_JJA_DIFF = np.average(data_p2['PRECT_JJA_BTALnEU'].data, axis=0) - np.average(data_p1['PRECT_JJA_BTALnEU'].data, axis=0)


# ========================= Calculate ttest ===============================

def cal_ttest(data1, data2):
    '''
        This function calculate difference using ttest
        data1 and data2 should be anomaly value compared with climatology
    '''
    import scipy.stats as stats

    # claim the array to save the p value
    p_array = np.zeros((data1.shape[1], data1.shape[2])) # it should be (lat, lon)

    # Compare on each point
    y_num = data1.shape[1]  ;  x_num = data1.shape[2]
    for yy in range(y_num):
        for xx in range(x_num):
            t_value, p_value = stats.ttest_ind(data1[:, yy, xx], data2[:, yy, xx])
            p_array[yy, xx] = p_value

    return p_array

def plot_diff_rainfall(diff_data, left_title, right_title, out_path, pic_name, p):
    '''This function plot the difference in precipitation'''
    # ------------ 2. Paint the Pic --------------------------
    from matplotlib import cm
    from matplotlib.colors import ListedColormap

    # 2.2 Set the figure
    proj    =  ccrs.PlateCarree()
    fig, ax =  plt.subplots(figsize=(15, 14), subplot_kw={'projection': proj})

#    # --- Set range ---
#    lonmin,lonmax,latmin,latmax  =  65,93,5,35
#    extent     =  [lonmin,lonmax,latmin,latmax]
#
#    # --- Tick setting ---
#    set_cartopy_tick(ax=ax,extent=extent,xticks=np.linspace(70,90,3,dtype=int),yticks=np.linspace(0,40,5,dtype=int),nx=1,ny=1,labelsize=15)

    # --- Set range ---
    lonmin,lonmax,latmin,latmax  =  65,93,5,35
    extent     =  [lonmin,lonmax,latmin,latmax]

    # --- Tick setting ---
    set_cartopy_tick(ax=ax,extent=extent,xticks=np.linspace(70,90,3,dtype=int),yticks=np.linspace(0,40,5,dtype=int),nx=1,ny=1,labelsize=15)

    # Shading for precipitation trend
    im  =  ax.contourf(lon, lat, diff_data, levels=levels, cmap='bwr_r', alpha=1, extend='both')

#    # Stippling picture
    sp  =  ax.contourf(lon, lat, p, levels=[0., 0.1], colors='none', hatches=['..'])

    # --- Coast Line ---
    ax.coastlines(resolution='50m', lw=1.5)

    # --- add patch for key area ---
    ax.add_patch(mpatches.Rectangle(xy=[74, 18], width=12, height=10,linewidth=1.5,
                                facecolor='none', edgecolor='orange',
                                transform=ccrs.PlateCarree()))

    # --- title ---
    ax.set_title(left_title, loc='left', fontsize=17.5)
    ax.set_title(right_title, loc='right', fontsize=17.5)

    # ========= add colorbar =================
    fig.subplots_adjust(top=0.8) 
    cbar_ax = fig.add_axes([0.2, 0.05, 0.6, 0.03]) 
    cb  =  fig.colorbar(im, cax=cbar_ax, shrink=0.5, pad=0.01, orientation='horizontal')
    #cb.ax.set_xticks(levels)
    cb.ax.tick_params(labelsize=15,)

    plt.savefig(out_path + pic_name)
    
# ================== Calculation fot ttest ========================================

# Prepare for the anomaly array
#anomaly_periodA = f0_p1['JJA_PRECT'].data.copy() ; anomaly_periodB = f0_p2['JJA_PRECT'].data.copy()

p_value_BTAL         = np.ones((len(lat), len(lon)))
p_value_BTALnEU      = np.ones((len(lat), len(lon)))
p_value_BTAL_BTALnEU = np.ones((len(lat), len(lon)))

for yy in range(len(lat)):
    for xx in range(len(lon)):
        anomaly_periodA_BTAL = data_p1['PRECT_JJA_BTAL'].data[:, yy, xx] - np.average(data['PRECT_JJA_BTAL'].data[:, yy, xx])
        anomaly_periodB_BTAL = data_p2['PRECT_JJA_BTAL'].data[:, yy, xx] - np.average(data['PRECT_JJA_BTAL'].data[:, yy, xx])
        a,b  = stats.ttest_ind(anomaly_periodA_BTAL, anomaly_periodB_BTAL, equal_var=False)
        p_value_BTAL[yy, xx] = b

        anomaly_periodA_BTALnEU = data_p1['PRECT_JJA_BTALnEU'].data[:, yy, xx] - np.average(data['PRECT_JJA_BTALnEU'].data[:, yy, xx])
        anomaly_periodB_BTALnEU = data_p2['PRECT_JJA_BTALnEU'].data[:, yy, xx] - np.average(data['PRECT_JJA_BTALnEU'].data[:, yy, xx])
        a,b  = stats.ttest_ind(anomaly_periodA_BTALnEU, anomaly_periodB_BTALnEU, equal_var=False)
        p_value_BTALnEU[yy, xx] = b

        anomaly_periodC_BTAL    = data_p3['PRECT_JJA_BTAL'].data[:, yy, xx] - np.average(data['PRECT_JJA_BTAL'].data[:, yy, xx])
        anomaly_periodC_BTALnEU = data_p3['PRECT_JJA_BTALnEU'].data[:, yy, xx] - np.average(data['PRECT_JJA_BTALnEU'].data[:, yy, xx])
        a,b  = stats.ttest_ind(anomaly_periodC_BTAL, anomaly_periodC_BTALnEU, equal_var=False)
        p_value_BTAL_BTALnEU[yy, xx] = b



def main():
    out_path = '/exports/csce/datastore/geos/users/s2618078/paint/analysis_EU_aerosol_climate_effect/ERL/'
    #plot_diff_rainfall(diff_data=prect_BTAL_JJA_DIFF, left_title='BTAL', right_title='JJA', out_path=out_path, pic_name="Aerosol_research_CESM_prect_BTAL_JJA_period_difference_1900_1960_231221.pdf")
    #plot_diff_rainfall(diff_data=prect_BTALnEU_JJA_DIFF,  left_title='BTALnEU', right_title='JJA',  out_path=out_path, pic_name="Aerosol_research_CESM_prect_BTALnEU_JJA_period_difference_1900_1960_231221.pdf")
    plot_diff_rainfall(diff_data=prect_BTAL_JJA_DIFF, left_title='(c)', right_title='CESM_ALL (JJA)', out_path=out_path, pic_name="ERL_fig1c_CESM_prect_BTAL_JJA_period_difference_1900_1960_231221.pdf", p=p_value_BTAL)
    plot_diff_rainfall(diff_data=prect_BTALnEU_JJA_DIFF, left_title='(b)', right_title='CESM_noEU (JJA)', out_path=out_path, pic_name="Aerosol_research_CESM_prect_BTALnEU_JJA_period_difference_1900_1960_231221.pdf", p=p_value_BTALnEU)
    plot_diff_rainfall(diff_data=(prect_BTAL_JJA_DIFF - prect_BTALnEU_JJA_DIFF), left_title='(d)', right_title='CESM_ALL - CESM_noEU (JJA)', out_path=out_path, pic_name="ERL_fig1d_CESM_prect_BTAL_sub_BTALnEU_JJA_period_difference_1900_1960_231227.pdf", p=p_value_BTAL_BTALnEU)


if __name__ == '__main__':
    main()