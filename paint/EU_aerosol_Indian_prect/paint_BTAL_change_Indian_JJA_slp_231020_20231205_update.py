'''
2023-10-18
This script calculate and paint 3D wind changes between two given periods
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

# select two period to calculate difference
class period:
    ''' This class infludes the two periods for compare '''
    periodA_1 = 50 ; periodA_2 = 70
    periodB_1 = 90 ; periodB_2 = 110

def archive_interpolated_model_output():
    '''
        Because I interpolated the model output data and save them in the same path,
        which is inconvient for CDO CAT, so this function aims to archive them in the different folder
    '''
    exp_name = ['BTAL', 'BTALnEU']
    var_name = ['OMEGA', 'RELHUM', 'T', 'U', 'V', 'Z3']

    all_files = os.listdir('/exports/csce/datastore/geos/groups/aerosol/sabineCESM/inp_sun2/')

    # archive folder structure expname/variable name/ensemble number
    for ii in exp_name:
        os.system('mkdir -p /exports/csce/datastore/geos/groups/aerosol/sabineCESM/inp_sun2/' + ii)
        for jj in var_name:
            os.system('mkdir -p /exports/csce/datastore/geos/groups/aerosol/sabineCESM/inp_sun2/' + ii + '/' + jj)
            path1  =  '/exports/csce/datastore/geos/groups/aerosol/sabineCESM/inp_sun2/' + ii + '/' + jj + '/'
            for ee in range(8): # ensemble members
                path2 = path1 + ii + '_' + str(ee + 1) + '/'
                os.system('mkdir -p ' + path2)
                for ff in all_files:
                    if '_'+ii+'_'+str(ee + 1) in ff and jj+'_B' in ff:
                        os.system('mv /exports/csce/datastore/geos/groups/aerosol/sabineCESM/inp_sun2/' + ff + ' ' + path2 + ff)
                        print('Successfully move the file ' + ff)

class calculate_class:
    out_path0 = '/exports/csce/datastore/geos/users/s2618078/data/model_data/SLP/'
    member_num = 8
    def cdo_ensemble(path1, exp_name, var_name, member_num):
        '''This function cdo the emsemble member result'''
        import os

        out_path1 = calculate_class.out_path0 + exp_name + '/'
        check_path(out_path1, 1)
        for i in range(member_num):
            path0 = path1 + exp_name + '/' + exp_name + '_' + str(i + 1) + '/' + 'mon/atm/' + var_name + '/'
            
            os.system('cdo cat ' + path0 + '*nc ' + out_path1 + exp_name + '_' + var_name + '_' + '1850_150years_member_' + str(i + 1) + '.nc')

    ensemble_path0 = '/exports/csce/datastore/geos/users/s2618078/data/model_data/SLP/'
    def cal_ensemble(exp_name, var_name):
        '''This function calculate the ensemble mean among all the members'''
        ensemble_path1 = calculate_class.ensemble_path0 + exp_name + '/' # This path saves the ncfiles
        file_list = os.listdir(ensemble_path1)
        ref_file  = xr.open_dataset(ensemble_path1 + file_list[0])

        # === Claim the array for result ===
        var_2d = np.zeros((1883, 96, 144))

        for i in range(calculate_class.member_num):
            f0 = xr.open_dataset(ensemble_path1 + exp_name + '_' + var_name + '_' + '1850_150years_member_' + str(i + 1) +'.nc')

            var_2d += f0[var_name].data/calculate_class.member_num

        print('Successfully calculated result for {} {}'.format(exp_name, var_name))

        # Write to nc file
        ncfile  =  xr.Dataset(
            {
                var_name: (["time", "lat", "lon"], var_2d),
            },
            coords={
                "time": (["time"], ref_file['time'].data),
                "lat":  (["lat"],  ref_file['lat'].data),
                "lon":  (["lon"],  ref_file['lon'].data),
            },
            )

        ncfile[var_name].attrs = ref_file[var_name].attrs

        ncfile.attrs['description']  =  'Created on 2023-10-20. This file is the ensemble mean among the 8 member in the {} emission experiments. The variable is {}.'.format(exp_name, var_name)
        ncfile.to_netcdf("/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/{}_{}_ensemble_mean_231020.nc".format(exp_name, var_name), format='NETCDF4')
        ncfile.attrs['script']       =  'paint_BTAL_change_Indian_JJA_slp_231020.py'

    def cal_jjas_mean(file_path, file_name, var_name, exp_name):
        '''This function calculate JJA mean for the data, which has been cdo cat and Ensemble_Mean'''
        file0 = xr.open_dataset(file_path + file_name)

        # === Claim 150 year array ===
        jjas_mean = np.zeros((157, 96, 144))

        data_JJAS = file0.sel(time=file0.time.dt.month.isin([6, 7, 8, 9]))

        for yyyy in range(157):
            jjas_mean[yyyy] = np.average(data_JJAS[var_name].data[yyyy * 4 : yyyy * 4 + 4], axis=0)
        
        print('{} {} JJAS mean calculation succeed!'.format(exp_name, var_name))

        # === Write to ncfile ===
        ncfile  =  xr.Dataset(
            {
                "{}_JJAS".format(var_name): (["time", "lat", "lon"], jjas_mean),
            },
            coords={
                "time": (["time"], np.linspace(1850, 1850+156, 157)),
                "lat":  (["lat"],  file0['lat'].data),
                "lon":  (["lon"],  file0['lon'].data),
            },
            )

        ncfile["{}_JJAS".format(var_name)].attrs = file0[var_name].attrs

        ncfile.attrs['description']  =  'Created on 2023-10-20, modified on 2023-12-05. This file is JJAS mean for the {}. Experiment is {}.'.format(var_name, exp_name)
        ncfile.attrs['script']       =  'paint_BTAL_change_Indian_JJA_slp_231020_20231205_update.py'
        ncfile.to_netcdf("/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/{}_{}_ensemble_mean_JJAS_231020.nc".format(var_name, exp_name), format='NETCDF4')
        

    def cal_two_periods_difference(data):
        ref_file    = xr.open_dataset('/exports/csce/datastore/geos/users/s2618078/data/model_data/WIND/BTAL/U/BTAL_U_1850_150years_member_1.nc')

        avg_periodA = np.average(data[period.periodA_1 : period.periodA_2], axis=0)
        avg_periodB = np.average(data[period.periodB_1 : period.periodB_2], axis=0)

        # Add coordination to the array
        ncfile  =  xr.Dataset(
            {
                "jja_diff": (["lev", "lat", "lon"], avg_periodB - avg_periodA),
            },
            coords={
                "lat":  (["lat"],  ref_file['lat'].data),
                "lon":  (["lon"],  ref_file['lon'].data),
                "lev":  (["lev"],  ref_file['lev'].data),
            },
            )
        
        return ncfile

class paint_class:
    '''This class save the settings and functions for painting'''
    # !!!!!!!!!! Settings !!!!!!!!!!!!
    # ======== Extent ==========
    lonmin,lonmax,latmin,latmax  =  30,135,-20,40
    extent     =  [lonmin,lonmax,latmin,latmax]

    # ======== Set Figure projection ==========
    proj       =  ccrs.PlateCarree()

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    def plot_uv_diff_at_select_lev(diff_u, diff_v, lev, plot_name):
        '''This function is to plot difference among two periods'''
        proj       =  ccrs.PlateCarree()

        ax         =  plt.subplot(projection=proj)

        # Extract level data
        u          =  diff_u.sel(lev=lev)['jja_diff'].data ; v          =  diff_v.sel(lev=lev)['jja_diff'].data

        # Tick settings
        cyclic_data_u, cyclic_lon = add_cyclic_point(u, coord=diff_u['lon'].data)
        cyclic_data_v, cyclic_lon = add_cyclic_point(v, coord=diff_u['lon'].data)
        set_cartopy_tick(ax=ax,extent=paint_class.extent,xticks=np.linspace(30,135,8,dtype=int),yticks=np.linspace(-20,60,5,dtype=int),nx=1,ny=1,labelsize=15)

        ax.coastlines(resolution='110m', lw=1.25)

        # Vector Map
        q  =  ax.quiver(cyclic_lon, diff_u['lat'].data, cyclic_data_u, cyclic_data_v, 
            regrid_shape=15, angles='uv',        # regrid_shape这个参数越小，是两门就越稀疏
            scale_units='xy', scale=0.05,        # scale是参考矢量，所以取得越大画出来的箭头就越短
            units='xy', width=0.55,              # width控制粗细
            transform=proj,
            color='k', headlength = 5, headaxislength = 4, headwidth = 4, alpha=0.8)

        ax.set_title('1901-1921 to 1941-1961',fontsize=17.5)

        add_vector_legend(ax=ax, q=q, speed=0.25)

        plt.savefig('/exports/csce/datastore/geos/users/s2618078/paint/analysis_EU_aerosol_climate_effect/wind/{}_period_diff_JJA_Indian_UV.pdf'.format(plot_name), dpi=500)

        ax.remove()

    def plot_uv_diff_at_select_lev_among_exps(diff_u1, diff_v1, diff_u2, diff_v2, lev,):
        '''This function is to plot difference among two periods'''
        proj       =  ccrs.PlateCarree()

        ax         =  plt.subplot(projection=proj)

        # Extract level data
        u1          =  diff_u1.sel(lev=lev)['jja_diff'].data ; v1         =  diff_v1.sel(lev=lev)['jja_diff'].data
        u2          =  diff_u2.sel(lev=lev)['jja_diff'].data ; v2         =  diff_v2.sel(lev=lev)['jja_diff'].data

        u           =  u1 - u2
        v           =  v1 - v2

        # Tick settings
        cyclic_data_u, cyclic_lon = add_cyclic_point(u, coord=diff_u1['lon'].data)
        cyclic_data_v, cyclic_lon = add_cyclic_point(v, coord=diff_u1['lon'].data)
        set_cartopy_tick(ax=ax,extent=paint_class.extent,xticks=np.linspace(30,135,8,dtype=int),yticks=np.linspace(-20,60,5,dtype=int),nx=1,ny=1,labelsize=15)

        ax.coastlines(resolution='110m', lw=1.25)

        # Vector Map
        q  =  ax.quiver(cyclic_lon, diff_u1['lat'].data, cyclic_data_u, cyclic_data_v, 
            regrid_shape=15, angles='uv',        # regrid_shape这个参数越小，是两门就越稀疏
            scale_units='xy', scale=0.03,        # scale是参考矢量，所以取得越大画出来的箭头就越短
            units='xy', width=0.5,              # width控制粗细
            transform=proj,
            color='k', headlength = 5, headaxislength = 4, headwidth = 4, alpha=0.8)

        ax.set_title('1901-1921 to 1941-1961',fontsize=17.5)

        add_vector_legend(ax=ax, q=q, speed=0.25)

        plt.savefig('/exports/csce/datastore/geos/users/s2618078/paint/analysis_EU_aerosol_climate_effect/wind/BTAL_BTALnEU_period_diff_JJA_Indian_UV.pdf', dpi=500)

        ax.remove()






def main():
    # === 1. Archive the intepolated modei output === !! Finished 2023-10-18-19:54!!
    #archive_interpolated_model_output()

    # === 2. Cdo cat the long-term files === !! Finished 2023-10-20-10:14!!
    src_path = '/exports/csce/datastore/geos/groups/aerosol/sabineCESM/'
    #calculate_class.cdo_ensemble(path1 = src_path, exp_name='BTAL', var_name='PSL', member_num=8)
    #calculate_class.cdo_ensemble(path1 = src_path, exp_name='BTALnEU', var_name='PSL', member_num=8)


    # === 3. Calculate ensemble mean among the members output === !! Finished 2023-10-20-10:21!!
    #calculate_class.cal_ensemble('BTAL', 'PSL')
    #calculate_class.cal_ensemble('BTALnEU', 'PSL')


    # === 4. Calculate JJA mean === !!Finished 2023-10-20-10:26!! 
    analysis_data_path = '/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/'
    calculate_class.cal_jjas_mean(file_path=analysis_data_path, 
                                file_name='BTAL_PSL_ensemble_mean_231020.nc', 
                                var_name='PSL',
                                exp_name='BTAL',
                                )
    calculate_class.cal_jjas_mean(file_path=analysis_data_path, 
                                file_name='BTALnEU_PSL_ensemble_mean_231020.nc', 
                                var_name='PSL',
                                exp_name='BTALnEU',
                                )
    # === Step 4 !!Finished 2023-10-19:14:26!!===

    # === 5. Calculate difference between two given period ===
    # This step is skipped, I calculate this in script paint_BTAL_change_Indian_JJA_wind_231018.py
#    
#    
#    # ================= 6. Painting section =========================================
#    # === 6.1 Period difference among the control experiment and noEU experiment themsleves ===
#    paint_class.plot_uv_diff_at_select_lev(con_diff_u, con_diff_v, 850, plot_name='BTAL')
#    paint_class.plot_uv_diff_at_select_lev(eu_diff_u,  eu_diff_v,  850, plot_name='BTALnEU')
#
#    paint_class.plot_uv_diff_at_select_lev_among_exps(con_diff_u,  con_diff_v, eu_diff_u, eu_diff_v, 850,)
#    #print(np.nanmin(con_diff_u['jja_diff']))



if __name__ == '__main__':
    main()