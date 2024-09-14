'''
2023-12-24
This script deals with the data from CESM BTAL/BTALnEU and GPCC, focusing on the variablity over the central Indian

The purpose is to provide dataset for bootstrap
'''
import xarray as xr
import numpy as np
import sys
import os
from scipy.stats import t
from scipy.stats import bootstrap
import matplotlib.pyplot as plt

sys.path.append('/exports/csce/datastore/geos/users/s2618078/uoe-code/module/')
from module_sun import cal_xydistance

import concurrent.futures

# ============================ File Location ==================================

file_CESM = xr.open_dataset("/exports/csce/datastore/geos/users/s2618078/data/analysis_data/CESM_PRECT_BTAL_BTALnEU_JJA_JJAS_1850_2006.nc")

file_GPCC = xr.open_dataset("/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/JJAS_GPCC_mean_update.nc")

key_area = [20, 28, 76, 87]
#print(file_CESM) # 1850 to 2006, 157 years
#print(file_GPCC) # 1891 to 2019, 129 years

# =============================================================================


# ===================== Calculate the 40year difference        ================

def cal_40year_difference(data, var_name, positive):
    #print(data)
    lat       = data.lat.data
    lon       = data.lon.data

    # 1. Extract the JJAS data
    if positive == 1: # South to North
        data_JJAS = data.sel(lat=slice(key_area[0], key_area[1]), lon=slice(key_area[2], key_area[3]))
    else:
        data_JJAS = data.sel(lat=slice(key_area[1], key_area[0]), lon=slice(key_area[2], key_area[3]))

    #print(data_JJAS)
    # 2. Claim the array to save the result
    num_year = int(len(data_JJAS.time.data))

    # 3. Calculate sample

    # 4. Calculate moving average
    data_JJAS_area_average = np.average(np.average(data_JJAS[var_name].data, axis=1), axis=1)
    #data_JJAS_area_average_moving = cal_moving_average(data_JJAS_area_average, 13)

    num_sample = len(data_JJAS_area_average) - 41
    #print(num_sample)
    sample     = np.zeros((num_sample,))

    for j in range(num_sample):
        sample[j] = np.average(data_JJAS[var_name].data[j + 40 ]) - np.average(data_JJAS[var_name].data[j ]) 

    return sample
    #return sample
# =============================================================================

# ========================== Bootstrap ========================================

def bootstrap(data):
    from scipy.stats import bootstrap
    rng = np.random.default_rng()

    data = (data,)

    res  = bootstrap(data, np.average, batch=1, n_resamples=5000, confidence_level=0.95, random_state=rng)

    return res

# =============================================================================

def get_cdf(data):
    n = len(data)

    x = np.sort(data)

    y = np.arange(1, 1+n)/n

    return x, y

def plot_ecdf(data,labelx,labely,title,color,out_path,out_name):
    """Plot ecdf"""
    import matplotlib.pyplot as plt
    # Call get_ecdf function and assign the returning values
    data = np.append(data, 0.4)
    x, y = get_cdf(data)
    
    #plt.plot(x,y,marker='.',linestyle='none',c='k')
    plt.plot(x,y,c='k')

    plt.xlabel(labelx)
    plt.ylabel(labely)
    plt.title(title)

    plt.xlim((-0.5, 0.4))
    plt.ylim((0, 1))

    plt.xticks([-0.4, -0.2, 0, 0.2, 0.4])
    plt.yticks([0, 0.25, 0.5, 0.75, 1], [0, 25, 50, 75, 100])
    
    plt.fill_between(x, y, color='lightblue', alpha=0.5)
    #plt.fill_between([np.max(x)-0.04, 0.4], [1, 1], color='lightblue', alpha=0.5)

    plt.plot([-0.7, -0.283], [0.025, 0.025], color='r')
    plt.plot([0.154, 0.7],   [0.975, 0.975], color='r')

    plt.plot([-0.283, -0.283], [0., 0.025], color='r')
    plt.plot([0.154, 0.154],   [0., 0.975], color='r')


    plt.savefig(out_path + out_name, dpi=650)

def cal_ttest(data):
    '''
        This function is to calculate the upper and bottom position of the data
    '''
        # 计算样本均值和标准差
    sample_mean = np.mean(data)
    print('The mean value is {}'.format(sample_mean))
    sample_std = np.std(data, ddof=1)
    print('The std value is {}'.format(sample_std))

    # 样本量和自由度
    n = len(data)
    df = n - 1
    print('df is {}'.format(df))

    # 95%置信水平的双边检验的alpha值
    alpha = 0.05  # 因为是双边，所以每边2.5%

    # 计算t临界值
    t_critical = t.ppf(1 - alpha, df)
    print('t_critical is {}'.format(t_critical))

    # 计算标准误差
    se = sample_std / np.sqrt(n)

    # 计算置信区间
    ci_lower = sample_mean - t_critical * se
    ci_upper = sample_mean + t_critical * se
    print('ci_lower is {}'.format(ci_lower))
    print('ci_upper is {}'.format(ci_upper))

    data = np.sort(data)
    print(data[43])
    position_lower = np.searchsorted(data, ci_lower, side='left')
    position_upper = np.searchsorted(data, ci_upper, side='left')
    print('lower is {}'.format(position_lower))
    print('upper is {}'.format(position_upper))

    return position_lower/len(data) * 100, position_upper/len(data) * 100

def cal_moving_average(x, w):
        return np.convolve(x, np.ones(w), "valid") / w

def cal_1900_1940_difference():
    file_CESM_p1 = file_CESM.sel(time=slice(1900, 1920), lat=slice(key_area[0], key_area[1]), lon=slice(key_area[2], key_area[3]))
    file_GPCC_p1 = file_GPCC.sel(time=slice(1900, 1920), lat=slice(key_area[1], key_area[0]), lon=slice(key_area[2], key_area[3]))

    file_CESM_p2 = file_CESM.sel(time=slice(1940, 1960), lat=slice(key_area[0], key_area[1]), lon=slice(key_area[2], key_area[3]))
    file_GPCC_p2 = file_GPCC.sel(time=slice(1940, 1960), lat=slice(key_area[1], key_area[0]), lon=slice(key_area[2], key_area[3]))

    return (np.average(file_GPCC_p2['JJAS_prect'].data) - np.average(file_GPCC_p1['JJAS_prect'].data)), (np.average(file_CESM_p2['PRECT_JJAS_BTAL'].data) - np.average(file_CESM_p1['PRECT_JJAS_BTAL'].data)), (np.average(file_CESM_p2['PRECT_JJAS_BTALnEU'].data) - np.average(file_CESM_p1['PRECT_JJAS_BTALnEU'].data))

def plot_boxplot(original_data, bootstrap_data, ):
    fig, axs = plt.subplots(figsize=(6, 6), )

    flierprops = dict(color='red', marker='o', markersize=0, linestyle='none', markeredgecolor='none')

    whiskerprops = dict(color='red', linestyle='-', linewidth=2)

    
#    axs.boxplot(original_data[0], positions=[0], whis=cal_ttest(original_data[0]), flierprops=flierprops, whiskerprops=whiskerprops, patch_artist=True, boxprops=dict(facecolor='lightgray', color='lightgray'), showmeans = True, meanline = True, capprops = dict(color = "red", linewidth = 2), widths=0.7, )
#    axs.boxplot(original_data[1], positions=[1], whis=cal_ttest(original_data[1]), flierprops=flierprops, whiskerprops=whiskerprops, patch_artist=True, boxprops=dict(facecolor='lightgray', color='lightgray'), showmeans = True, meanline = True, capprops = dict(color = "red", linewidth = 2), widths=0.7, )
#    axs.boxplot(original_data[2], positions=[2], whis=cal_ttest(original_data[2]), flierprops=flierprops, whiskerprops=whiskerprops, patch_artist=True, boxprops=dict(facecolor='lightgray', color='lightgray'), showmeans = True, meanline = True, capprops = dict(color = "red", linewidth = 2), widths=0.7, )

    axs.boxplot(original_data[0], positions=[0], whiskerprops=whiskerprops, patch_artist=True, boxprops=dict(facecolor='lightgray', color='lightgray'), showmeans = True, meanline = True, capprops = dict(color = "red", linewidth = 2), widths=0.7, bootstrap=9999)
    axs.boxplot(original_data[1], positions=[1], whiskerprops=whiskerprops, patch_artist=True, boxprops=dict(facecolor='lightgray', color='lightgray'), showmeans = True, meanline = True, capprops = dict(color = "red", linewidth = 2), widths=0.7, bootstrap=9999)
    axs.boxplot(original_data[2], positions=[2], whiskerprops=whiskerprops, patch_artist=True, boxprops=dict(facecolor='lightgray', color='lightgray'), showmeans = True, meanline = True, capprops = dict(color = "red", linewidth = 2), widths=0.7, bootstrap=9999)


#    gpcc, btal, btalneu = cal_1900_1940_difference()
#    axs.scatter(0, gpcc, marker='^', color='green', s=100, zorder=100)
#    axs.scatter(1, btal, marker='^', color='green',   s=100, zorder=100)
#    axs.scatter(2, btalneu, marker='^', color='green', s=100, zorder=100)
#
#    axs.set_yticks(np.linspace(-2.5, 2.5, 11))
#    axs.set_ylim((-2.5, 2.5))

    axs.set_xticklabels(["GPCC", "BTAL", "BTALnEU"])
    #axs.boxplot(bootstrap_data[0], positions=[1],)
    #axs.boxplot(bootstrap_data[1], positions=[2],)
    #axs.boxplot(bootstrap_data[2], positions=[3],)



    plt.savefig('test.png')

# ========================== main =============================================

def main():
    # For the cal_40year_difference

    CESM_prect_BTAL     = cal_40year_difference(file_CESM, 'PRECT_JJAS_BTAL', 1)
    CESM_prect_BTALnEU  = cal_40year_difference(file_CESM, 'PRECT_JJAS_BTALnEU', 1)

    GPCC_prect          = cal_40year_difference(file_GPCC, 'JJAS_prect', 0)

#    print(np.sort(CESM_prect_BTAL))
 #   print("sample calculation completed!")
    cal_ttest(GPCC_prect)

#    with concurrent.futures.ProcessPoolExecutor() as executor:
#        s1  =  executor.submit(bootstrap, CESM_prect_BTAL)
#        s2  =  executor.submit(bootstrap, CESM_prect_BTALnEU)
#        s3  =  executor.submit(bootstrap, GPCC_prect)
#
#        cesm_btal        =  s1.result()
#        cesm_btalneu     =  s2.result()
#        gpcc             =  s3.result()
#
#    print("Bootstrap completed!")



    # 创建箱线图
    #data = gpcc.bootstrap_distribution
    #data = [gpcc.bootstrap_distribution, cesm_btal.bootstrap_distribution, cesm_btalneu.bootstrap_distribution]
    # print(np.min(data)) ; print(np.max(data))
    #print(cal_ttest(GPCC_prect))
    plot_boxplot(original_data=[GPCC_prect, CESM_prect_BTAL, CESM_prect_BTALnEU], bootstrap_data=0)







if __name__ == '__main__': 
    main()