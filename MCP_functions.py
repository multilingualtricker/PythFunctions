#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 12:44:17 2017

@author: markprosser
"""
#******************************************************************************
#NB Python -> Preferences -> IPython console -> Graphics -> Automatic
def show_plot(figure_id=None):
    import matplotlib.pyplot as plt
    if figure_id is None:
        fig = plt.gcf()
    else:
        # do this even if figure_id == 0
        fig = plt.figure(num=figure_id)

    plt.show()
    plt.pause(1e-9)
    fig.canvas.manager.window.activateWindow()
    fig.canvas.manager.window.raise_()
    

#******************************************************************************
def get_IGCM(run, variable): #created 19oct2017
#e.g. get_IGCM(CONT, 8)
#retrieves data from IGCM for 1 variable
    import numpy as np
    from netCDF4 import Dataset
    
    if run == 'CONT':
        dataset = Dataset("/Users/markprosser/Desktop/PythonScripts/Manoj1/IGCMd.cdf") #IGCM_CONT
    elif run == 'LNC': 
        dataset = Dataset("/Users/markprosser/Desktop/PythonScripts/Manoj1/IGCMg.cdf") #IGCM_LNC

    l1_var = list(dataset.variables.keys())
    lname = l1_var[variable]
    lnamesize = dataset.variables[lname][:]
    dataset_1vara = np.full(lnamesize.shape,np.nan)
    dataset_1vara[:] = dataset.variables[lname][:]
    dataset_1var = np.squeeze(dataset_1vara, axis=1)
    
    var_name = [dataset.variables[lname].long_name, lname, dataset.variables[lname].units]

    return dataset_1var, var_name
#******************************************************************************
def get_IGCM650(run, variable): #created 19oct2017
#e.g. get_IGCM(CONT, 8)
#retrieves data from IGCM for 1 variable
#0='longitude',
#1='latitude',
#2='level',
#3='pressure',
#4='surface',
#5='zonal',
#6='time',
#7='ST',
#8='PTOT'

    import numpy as np
    from netCDF4 import Dataset
    
    if run == 'CONT':
        dataset1 = Dataset("/Users/markprosser/Desktop/PythonScripts/Manoj1/New650/NewCONT650/478830-572400-qTs.cdf") #IGCM_CONT
        dataset2 = Dataset("/Users/markprosser/Desktop/PythonScripts/Manoj1/New650/NewCONT650/572430-712800-qTs.cdf") #IGCM_CONT
    elif run == 'LNC': 
        dataset1 = Dataset("/Users/markprosser/Desktop/PythonScripts/Manoj1/New650/NewLNC650/478830-572400-qTs.cdf") #IGCM_CONT
        dataset2 = Dataset("/Users/markprosser/Desktop/PythonScripts/Manoj1/New650/NewLNC650/572430-712800-qTs.cdf") #IGCM_CONT
    elif run == 'LNCv':
        dataset1 = Dataset("/Users/markprosser/Desktop/PythonScripts/Manoj1/New650/NewLNCV650/478830-572400-qTs.cdf") #IGCM_CONT
        dataset2 = Dataset("/Users/markprosser/Desktop/PythonScripts/Manoj1/New650/NewLNCV650/572430-712800-qTs.cdf") #IGCM_CONT


    l1_var1 = list(dataset1.variables.keys())
    lname1 = l1_var1[variable]
    lnamesize1 = dataset1.variables[lname1][:]
    dataset_1vara1 = np.full(lnamesize1.shape,np.nan)
    dataset_1vara1[:] = dataset1.variables[lname1][:]
    dataset_1var1 = np.squeeze(dataset_1vara1, axis=1)

    l1_var2 = list(dataset2.variables.keys())
    lname2 = l1_var2[variable]
    lnamesize2 = dataset2.variables[lname2][:]
    dataset_1vara2 = np.full(lnamesize2.shape,np.nan)
    dataset_1vara2[:] = dataset2.variables[lname2][:]
    dataset_1var2 = np.squeeze(dataset_1vara2, axis=1)

    dataset_1var =  np.concatenate((dataset_1var1, dataset_1var2), axis=0)

    var_name = [dataset1.variables[lname1].long_name, lname1, dataset1.variables[lname1].units]

    return dataset_1var, var_name

#******************************************************************************
def get_IGCM650xr(run, var): #created 26may2018
    #e.g. get_IGCM(CONT, 'ST')
    #retrieves data from IGCM for 1 variable
    #0='longitude',
    #1='latitude',
    #2='level',
    #3='pressure',
    #4='surface',
    #5='zonal',
    #6='time',
    #7='ST',
    #8='PTOT'

    import numpy as np
    from netCDF4 import Dataset
    import xarray as xr

    if run == 'CONT':
        dataset1 = xr.open_dataset("/Users/markprosser/Desktop/PythonScripts/Manoj1/New650/NewCONT650/478830-572400-qTs.cdf") #IGCM_CONT
        dataset2 = xr.open_dataset("/Users/markprosser/Desktop/PythonScripts/Manoj1/New650/NewCONT650/572430-712800-qTs.cdf") #IGCM_CONT
    elif run == 'LNC': 
        dataset1 = xr.open_dataset("/Users/markprosser/Desktop/PythonScripts/Manoj1/New650/NewLNC650/478830-572400-qTs.cdf") #IGCM_CONT
        dataset2 = xr.open_dataset("/Users/markprosser/Desktop/PythonScripts/Manoj1/New650/NewLNC650/572430-712800-qTs.cdf") #IGCM_CONT
    elif run == 'LNCv':
        dataset1 = xr.open_dataset("/Users/markprosser/Desktop/PythonScripts/Manoj1/New650/NewLNCV650/478830-572400-qTs.cdf") #IGCM_CONT
        dataset2 = xr.open_dataset("/Users/markprosser/Desktop/PythonScripts/Manoj1/New650/NewLNCV650/572430-712800-qTs.cdf") #IGCM_CONT

    abc = dataset1.squeeze(dim='surface')
    abcB = abc[var]
    abc2 = dataset2.squeeze(dim='surface')
    abc2B = abc2[var]
    abc3 = xr.concat([abcB, abc2B], dim='time')

    return abc3
#******************************************************************************
def yearly_average_over_3months(dataset, ml):
    #assumes dataset[0]=Jan, dataset[11]=Dec
    import numpy as np
    abc = np.full((int(dataset.shape[0]/12), dataset.shape[1], dataset.shape[2]), np.nan)
    abc2 = abc
    for i in range(0, abc.shape[0]):
        abc2[i,:,:] = (dataset[(ml[0]-1)+(i*12),:,:] + dataset[(ml[1]-1)+(i*12),:,:] + dataset[(ml[2]-1)+(i*12),:,:])
        #abc3 = abc2
        yearly_data = abc2 / 3
    return yearly_data


#******************************************************************************
def get_ocean():
    return 2**.5


#******************************************************************************


import numpy as np
from scipy import stats
from itertools import combinations
from statsmodels.stats.multitest import multipletests
#from statsmodels.stats.libqsturng import psturng
import warnings


def kw_dunn(groups, to_compare=None, alpha=0.05, method='bonf'):
    """
    Kruskal-Wallis 1-way ANOVA with Dunn's multiple comparison test
    Arguments:
    ---------------
    groups: sequence
        arrays corresponding to k mutually independent samples from
        continuous populations
    to_compare: sequence
        tuples specifying the indices of pairs of groups to compare, e.g.
        [(0, 1), (0, 2)] would compare group 0 with 1 & 2. by default, all
        possible pairwise comparisons between groups are performed.
    alpha: float
        family-wise error rate used for correcting for multiple comparisons
        (see statsmodels.stats.multitest.multipletests for details)
    method: string
        method used to adjust p-values to account for multiple corrections (see
        statsmodels.stats.multitest.multipletests for options)
    Returns:
    ---------------
    H: float
        Kruskal-Wallis H-statistic
    p_omnibus: float
        p-value corresponding to the global null hypothesis that the medians of
        the groups are all equal
    Z_pairs: float array
        Z-scores computed for the absolute difference in mean ranks for each
        pairwise comparison
    p_corrected: float array
        corrected p-values for each pairwise comparison, corresponding to the
        null hypothesis that the pair of groups has equal medians. note that
        these are only meaningful if the global null hypothesis is rejected.
    reject: bool array
        True for pairs where the null hypothesis can be rejected for the given
        alpha
    Reference:
    ---------------
    Gibbons, J. D., & Chakraborti, S. (2011). Nonparametric Statistical
    Inference (5th ed., pp. 353-357). Boca Raton, FL: Chapman & Hall.
    """

    # omnibus test (K-W ANOVA)
    # -------------------------------------------------------------------------

    groups = [np.array(gg) for gg in groups]

    k = len(groups)

    n = np.array([len(gg) for gg in groups])
    if np.any(n < 5):
        warnings.warn("Sample sizes < 5 are not recommended (K-W test assumes "
                      "a chi square distribution)")

    allgroups = np.concatenate(groups)
    N = len(allgroups)
    ranked = stats.rankdata(allgroups)

    # correction factor for ties
    T = stats.tiecorrect(ranked)
    if T == 0:
        raise ValueError('All numbers are identical in kruskal')

    # sum of ranks for each group
    j = np.insert(np.cumsum(n), 0, 0)
    R = np.empty(k, dtype=np.float)
    for ii in range(k):
        R[ii] = ranked[j[ii]:j[ii + 1]].sum()

    # the Kruskal-Wallis H-statistic
    H = (12. / (N * (N + 1.))) * ((R ** 2.) / n).sum() - 3 * (N + 1)

    # apply correction factor for ties
    H /= T

    df_omnibus = k - 1
    p_omnibus = stats.chisqprob(H, df_omnibus)

    # multiple comparisons
    # -------------------------------------------------------------------------

    # by default we compare every possible pair of groups
    if to_compare is None:
        to_compare = tuple(combinations(range(k), 2))

    ncomp = len(to_compare)

    Z_pairs = np.empty(ncomp, dtype=np.float)
    p_uncorrected = np.empty(ncomp, dtype=np.float)
    Rmean = R / n

    for pp, (ii, jj) in enumerate(to_compare):

        # standardized score
        Zij = (np.abs(Rmean[ii] - Rmean[jj]) /
               np.sqrt((1. / 12.) * N * (N + 1) * (1. / n[ii] + 1. / n[jj])))
        Z_pairs[pp] = Zij

    # corresponding p-values obtained from upper quantiles of the standard
    # normal distribution
    p_uncorrected = stats.norm.sf(Z_pairs) * 2.

    # correction for multiple comparisons
    reject, p_corrected, alphac_sidak, alphac_bonf = multipletests(
        p_uncorrected, method=method
    )

    return H, p_omnibus, Z_pairs, p_corrected, reject

#******************************************************************************
def yearly_average_n_months(monthly_data, n, start_month):
    #this function iterates through the monthly data set looking for the 1st instance of (e.g.) a JAN (% 12 == 0), \
    #once it finds it, it then sums the subsequent n months then divides by n to get the average for that series of
    #of months, it avoids the last year in the data set as this causes range issues/errors when you have months from 2 
    #different years. e.g. DJF
    
    yearly_data = np.full(((int(monthly_data.shape[0]/12)-1), monthly_data.shape[1], \
                           monthly_data.shape[2]), 0)

    index_check = []

    for i in range(0, yearly_data.shape[0]):
        for j in range(0, 12):
            if ((i*12)+j) % 12 == start_month:
                for k in range(0, n):
                    #print((i*12)+j+k)
                    index_check.append((i*12)+j+k)
                    yearly_data[i,:,:] += monthly_data[((i*12)+j+k),:,:]


    yearly_data /= 3

    return yearly_data, index_check

#******************************************************************************
    
def running_mean_3d(yearly_data, n): 
    import random
    #calculates a running mean for 3d array data e.g. [year=390, lat=64, long=128]

    rm_yearly_data = np.full((yearly_data.shape[0], yearly_data.shape[1], yearly_data.shape[2]), np.nan)
    rm_yearly_data.shape

    if n % 2 != 0:
        mid = int(np.floor(n/2))
        d_mid = int(np.floor(n/2))

        for i in range(mid, rm_yearly_data.shape[0]-mid):
            rm_yearly_data[i,:,:] = np.mean(yearly_data[i-d_mid:i+d_mid+1,:,:],0)        
    else:
        mid = int(n/2) 
        d_mid = int(n/2) 
    #         #print(i)
    #         # even n yet to be implementedyet to be implemented
    
    
    
    #FUNCTION TEST
    randy = random.randint(0, yearly_data.shape[1]-1)
    randx = random.randint(0, yearly_data.shape[2]-1)

    randt = random.randint(mid, rm_yearly_data.shape[0]-mid)
    a = round(rm_yearly_data[randt, randy, randx], 9)
    b = round(np.mean(yearly_data[randt-d_mid:randt+d_mid+1, randy, randx]), 9)

    print(a)
    print(b)

    if a != b:
        raise ValueError('FUNCTION RANDOM TEST FAILED')
    else:
        print('function random test passed')
        
    rm_yearly_data = rm_yearly_data[mid:rm_yearly_data.shape[0]-mid,:,:]
 
    return rm_yearly_data

#******************************************************************************
def gen_3d_manoj_land_mask(n):
    #generates an [n x 64 x 128] landmask
    import io
    import numpy as np
    import numpy.ma as ma
    with io.open('/Users/markprosser/Desktop/PythonScripts/Input_needed_by/Manoj1_9/Mask_present_128_64', "r", encoding='utf-8-sig') as f:
        data = f.read()
    data2 = data.split('      ')
    data3 = data2[1:len(data2)]

    mask = np.full((128,64), np.nan)
    for j in range(0,64):
        for i in range(0,128):
            mask[i, j] = data3[(j*128)+i]

    mask2 = np.rot90(mask, 3)
    mask3 = np.fliplr(mask2)
    mask4 = ma.array(mask3) #now a masked array

    mask_3D = np.repeat(mask4[np.newaxis, :, :], n, axis=0)
    #mask_3D.dump('/Users/markprosser/Desktop/PythonScripts/Input_needed_by/Manoj1_9/Mask_3D_389')
    
    return mask_3D    

#******************************************************************************
def process_calc_array2python_array(data):
    #once you've got a calc array into python it won't be immediately usable for plotting, this function will process it.
    
    #strip out \n
    data2 = [x.strip('\n') for x in data]
    #strip out beginning [[ and end ]]
    data3 = [x.replace('[[', ' [') for x in data2]
    data4 = [x.replace(']]', '] ') for x in data3]
    #strip out gap '[ ' at the beginning
    data5 = [x.replace(' [', '[') for x in data4]
    #strip out blank at the beginning and end
    data6 = [x.replace('[ ', '[') for x in data5]
    data7 = [x.replace(' ]', ']') for x in data6]
    #put in commas
    data8 = [x.replace(' ', ',') for x in data7]
    #get rid of [ and ]
    data9 = [x.strip('[') for x in data8]
    data10 = [x.strip(']') for x in data9]
    #convert from string to list
    data11 = [x.split(",") for x in data10]

    array = np.full((len(data11),len(data11[0])),np.nan)
    array.shape

    for y in range(0, len(data11)):
        for x in range(0, len(data11[0])):
            try:
                array[y][x] = float(data11[y][x])
            except Exception:
                pass

    return(array)   
#******************************************************************************
def get_KNMI(run, variable):

    from netCDF4 import Dataset

    if run == 'tos_Omon_modmean_rcp85_000.nc':
        dataset = Dataset("/Users/markprosser/Desktop/PythonScripts/Manoj1/KNMI/tos_Omon_modmean_rcp85_000.nc") 
    elif run == 'tas_Amon_modmean_piControl_000': 
        dataset = Dataset("/Users/markprosser/Desktop/PythonScripts/Manoj1/KNMI/tas_Amon_modmean_piControl_000.nc") 

    list_vars = list(dataset.variables.keys()) #shows which variables
    var_name = list_vars[variable] #e.g. tas
    var_data = dataset.variables[var_name][:]
    try:
        var_details = [dataset.variables[var_name].long_name, var_name, \
                   dataset.variables[var_name].units]
    except Exception:
        var_details = np.nan
    
    return var_data, var_details
#******************************************************************************
def get_KNMImask(depth):
    from netCDF4 import Dataset
    dataset = Dataset("/Users/markprosser/Desktop/PythonScripts/Manoj1/KNMI/lsmask_cmip3_144(1).nc")
    mask_2D = dataset.variables['lsmask'][:]
    mask_3D = np.repeat(mask_2D[np.newaxis, :, :], depth, axis=0)
    return mask_2D, mask_3D
    #144*72
#******************************************************************************
def monthly_anomalies(data):
    #stick in a monthly dataset, it will return the monthly anomalies of it.
    monthly_av = np.full((12, data.shape[1], data.shape[2]), 0)
    for i in range(0, int(data.shape[0]/12)):
        for j in range(0,12):
            monthly_av[j,:,:] = monthly_av[j,:,:] + data[(i*12)+j,:,:]
    monthly_av = monthly_av/(data.shape[0]/12)

    monthly_anom = np.full((data.shape[0], data.shape[1], data.shape[2]), 0)
    for i in range(0, int(data.shape[0])):
        monthly_anom[i,:,:] = data[i,:,:] - monthly_av[(i%12),:,:]
    check = np.sum(np.mean(monthly_anom, axis=0))
    #check should be very close to zero if the function is correct
    return check, monthly_anom
#******************************************************************************
def SN_get_geog_area(global_data, lats, longs, lat_N, lat_S, long_W, long_E):
    #use if lats go [-90.....+90] like KNMI's
    #and if global_data[0,0,:]=S.Pole - upside down map
    lat_Nt = int((lat_N+90)*(global_data.shape[1]/180))
    lat_St = int((lat_S+90)*(global_data.shape[1]/180))
    lats_t = lats[lat_St:lat_Nt]

    long_Wt = int(np.rint(long_W*(global_data.shape[2]/360)))
    long_Et = int(np.rint(long_E*(global_data.shape[2]/360)))
    longs_t = longs[long_Wt:long_Et]

    sub_data = global_data[:,lat_St:lat_Nt,long_Wt:long_Et]
    return sub_data, lats_t, longs_t
#******************************************************************************
def NS_get_geog_area(global_data, lats, longs, lat_N, lat_S, long_W, long_E):
    #use if lats go [+90.....-90] like Manoj's
    #and if global_data[0,0,:]=N.pole
    lat_Nt = int(np.rint(abs((lat_N-90)*(global_data.shape[1]/180))))
    lat_St = int(np.rint(abs((lat_S-90)*(global_data.shape[1]/180))))
    lats_t = lats[lat_Nt:lat_St]

    long_Wt = int(np.rint(long_W*(global_data.shape[2]/360)))
    long_Et = int(np.rint(long_E*(global_data.shape[2]/360)))
    longs_t = longs[long_Wt:long_Et]

    sub_data = global_data[:,lat_Nt:lat_St,long_Wt:long_Et]
    return sub_data, lats_t, longs_t    
#******************************************************************************
def bespoke_global_mask(deglat, deglong, depth):
    from scipy import interpolate
    from netCDF4 import Dataset
    import numpy.ma as ma
    dataset = Dataset("/Users/markprosser/Desktop/PythonScripts/Manoj1/NCEP/OTHER/MASK13.nc")
    highres_mask = dataset.variables['lsmask'][:]
    if highres_mask.shape[0] == 1:
        highres_mask = np.squeeze(highres_mask, axis=0)
    nlat = 180/deglat 
    nlong = 360/deglong
    half_latinterval = (highres_mask.shape[0]/nlat) / 2
    half_longinterval = (highres_mask.shape[1]/nlong) / 2

    #stage you don't really understand, but necessary
    x = np.arange(0, highres_mask.shape[0], 1)
    y = np.arange(0, highres_mask.shape[1], 1)
    xx, yy = np.meshgrid(x, y)
    f = interpolate.interp2d(y, x, highres_mask, kind='cubic')

    #the business
    Ynew = np.linspace(half_latinterval, highres_mask.shape[0]-half_latinterval, nlat) #the lat points at which you want to interpolate
    Xnew = np.linspace(half_longinterval, highres_mask.shape[1]-half_longinterval, nlong) #the long points at which you want to interpolate
    interpolated_mask = np.rint(f(Xnew, Ynew)) #interpolation
    
    interpolated_mask2 = np.flip(interpolated_mask,0) #now upside down
    interpolated_mask_ma = ma.array(interpolated_mask2)
    
    
    mask_3D = np.repeat(interpolated_mask_ma[np.newaxis, :, :], depth, axis=0)
    
    return mask_3D   
#******************************************************************************
def get_NCEP(run, variable):
    from netCDF4 import Dataset
    if run == 'air.sig995.mon.mean.nc':
        dataset = Dataset("/Users/markprosser/Desktop/PythonScripts/Manoj1/NCEP/air.sig995.mon.mean.nc") 
        #elif run == 'tas_Amon_modmean_piControl_000': 
#            dataset = Dataset("/Users/markprosser/Desktop/PythonScripts/Manoj1/KNMI/tas_Amon_modmean_piControl_000.nc")
    list_vars = list(dataset.variables.keys()) #shows which variables
    var_name = list_vars[variable] #e.g. tas
    var_data = dataset.variables[var_name][:]
    try:
        var_details = [dataset.variables[var_name].long_name, var_name, \
                       dataset.variables[var_name].units]
    except Exception:
        var_details = np.nan
    
    return var_data, var_details
#******************************************************************************
def dictmodno(input3Darray, modno):
    #created 26may2018
    #input3Darray is 3D array e.g. [time, lat, long]
    #modno e.g. 12 if you want to put all the jans, febs together together etc
    #KW - mod, jans, febs, dictionary

    nanmat = np.full((int(input3Darray.shape[0]/modno), input3Darray.shape[1],input3Darray.shape[2]), np.nan)
    D = {k:np.copy(nanmat) for k in range(0,modno)}

    for i in range(0,input3Darray.shape[0]):
        D[i%modno][int(np.floor(i/modno)),:,:] = input3Darray[i,:,:]

    return D
#******************************************************************************
def monthly2yearlyaverage(input3Darray, N):
    #input3Darray e.g. [time, lat, long]
    #N = 12 if you're doing the standard jan-dec

    nanmat = np.full((int(input3Darray.shape[0]/N), input3Darray.shape[1], input3Darray.shape[2]), np.nan)

    for i in range(0, int(input3Darray.shape[0]/N)):
        abc = input3Darray[(i*N):(i*N)+N,:,:]
        abc2 = np.nanmean(abc, axis=0)
        nanmat[i,:,:] = abc2

    return nanmat
#******************************************************************************
def countnans(inputarray):
    #KW - count nans
    abc = np.count_nonzero(np.isnan(inputarray))
    print('number of nans = ' +str(abc))
    tot=1
    for i in range(0,len(inputarray.shape)):
        tot *= inputarray.shape[i]
    print(inputarray.shape)
    print('number of total elements = ' +str(tot))  
    abc2 = abc/tot*100
    print('proportion of nans roughly ' +str(np.round(abc2)) + '%')  
#******************************************************************************
def cellmidpoints(nlat, nlong):
    #KW - midpoint latitude longitude
    latdeg = 180/nlat
    longdeg = 360/nlong
    lats = np.arange(90-(latdeg/2),-90,-latdeg)
    longs = np.arange(0+(longdeg/2), 360, longdeg)
    return lats, longs
#******************************************************************************
def geogareaweighting2D(nlat, nlong, bespokelatmidpoints, bespokelongmidpoints, optA):
    #lat = number of cells e.g. 64
    #bespokelatmidpoints = 0 #is standard, if you want to use bespoke lats, put in array here!
    #optA=0 - cells all have value of 1
    #optA=1 - cells at equator have an area of 1
    #optA=2 - cells have values of their area in km^2
    #optA=3 - cells have value = cos(latmidpoint)
    import MCP_functions as MCP
    latdegheight = 180/nlat #in degrees
    d = latdegheight/2
    nanmat = np.full((nlat,1), np.nan)
    if type(bespokelatmidpoints) == int:
        lats1, longs1 = MCP.cellmidpoints(nlat, nlong)
    else:
        lats1 = np.copy(bespokelatmidpoints)
        longs1 = np.copy(bespokelongmidpoints)
    if optA == 0:
        for i in range(lats1.shape[0]):
            nanmat[i,0] = 1
    elif optA == 1:
        for i in range(lats1.shape[0]):
            nanmat[i,0] = (np.sin(np.deg2rad(lats1[i]+d))-np.sin(np.deg2rad(lats1[i]-d))) / np.sin(np.deg2rad(2*d))
    elif optA == 2:
        for i in range(lats1.shape[0]):
            nanmat[i,0] = np.sin(np.deg2rad(lats1[i]+d))-np.sin(np.deg2rad(lats1[i]-d))
    elif optA == 3:
        for i in range(lats1.shape[0]):
            nanmat[i,0] = np.cos(np.deg2rad(lats1[i]))
   
    finalmat = np.tile(nanmat, nlong)

    return finalmat
#******************************************************************************
def getweighted_climatemodelmean(array, weightings, optA):
    #masked (or not) array in 3D e.g. shape (650, 64, 128)
    #masked (or not) weightings in 2D e.g. shape (64, 128)
    #optA=1 = element nanmean
    #optA=2 = axis method nanmean
    nansum2d = np.nansum(weightings)
    nansum3d = nansum2d * array.shape[0]

    if optA == 1: #element nanmean
        weighted_array = array * weightings
        abc = np.nansum(weighted_array)
        answer = abc / nansum3d

    elif optA == 2: #axis nanmean
        time_nmean = np.nanmean(array,0) 
        weighted_array = time_nmean * weightings
        abc = np.nansum(weighted_array) 
        answer = abc / nansum2d


    return answer  
#******************************************************************************
def running_mean_1d(VEC,rm):
    #created 8nov2016
    #VEC - column vector (e.g. (650, 1)) you want to operate on
    #rm - e.g. 3 or 5 - no even numbers
    import numpy as np
    side = int((rm-1)/2)
    z = np.empty((VEC.shape[0],1))
    z[:]=np.NAN
    for i in range(side,VEC.shape[0]-side):
        z[i,0]=np.average(VEC[i-side:i+side+1,0])
    return z
#******************************************************************************
def ocean_basin_mask(interp_mask, nlat, nlong, depth, optA, optB):
    #interp mask can be shape (1, 64, 128) or (64, 128)
    #optA
        #1 = Atlantic left
        #2 = Indian left
        #3 = Pacific left
        #4 = Southern Ocean
        #5 = Arctic Ocean
    #optB
        #1 = Whole left
        #2 = North left
        #3 = South left
    import numpy.ma as ma
    if len(interp_mask.shape) == 3:
        interp_mask = np.squeeze(interp_mask)
    interp_mask = np.flip(interp_mask,0)
    if optA != 0: #if function isn't turned off
        interp_mask2 = np.tile(interp_mask, 2)
    
        halflong = int(round(interp_mask2.shape[1]/2))
        halflat = int(round(interp_mask2.shape[0]/2))
        S2 = int(round(interp_mask2.shape[1]))
        S1 = int(round(interp_mask2.shape[0]))
    
        if optA == 1: #Atlantic
            interp_mask2[:int(round(((120.263/180)*S1))), (round((19.998/360)*halflong)+halflong):] = 0
            #Gets rid of east of tip of South Africa
            interp_mask2[int(round(((137.313/180)*S1))):, (round((16.615/360)*halflong)+halflong):] = 0
            #Gets rid of east of tip of Svalbard
            interp_mask2[:S1, (round((41.823/360)*halflong)+halflong):] = 0
            #Gets rid of east of the East of the Black Sea
            interp_mask2[:int(round(((98.919/180)*S1))), :(round((290.858/360)*halflong))] = 0
            #Gets rid of west of tip of South America
            interp_mask2[int(round(((156.563/180)*S1))):, :] = 0
            #Gets rid of north of Arctic circle
            interp_mask2[:, 0:(round((262.101/360)*halflong))] = 0
            #Gets rid of west of gulf of Mexico
            interp_mask2[:int(round(((108.148/180)*S1))), 0:(round((271.054/360)*halflong))] = 0
            #Awkward corner around central America 1
            interp_mask2[:int(round(((105.681/180)*S1))), 0:(round((276.124/360)*halflong))] = 0
            #Awkward corner around central America 2
            interp_mask2[:int(round(((30.000/180)*S1))), :] = 0
            #Gets rid of Southern Ocean
            if optB == 2: #North Atlantic remaining
                interp_mask2[:int(round(((90.000/180)*S1))), :] = 0
            if optB == 3: #South Atlantic remaining
                interp_mask2[int(round(((90.000/180)*S1))):, :] = 0
        elif optA == 2: #Indian 
            interp_mask2[:int(round(((120.527/180)*S1))), :(round((19.998/360)*halflong)+halflong)] = 0
            #Gets rid of west of tip of South Africa
            interp_mask2[:int(round(((30.000/180)*S1))), :] = 0
            #Gets rid of Southern Ocean
            interp_mask2[int(round(((120.527/180)*S1))):, :] = 0
            #Gets rid of north of Persian Gulf
            interp_mask2[:, (round((143.511/360)*halflong)+halflong):] = 0
            #Gets rid of east of Tasmania Australia
            interp_mask2[int(round(((60.000/180)*S1))):, (round((130.247/360)*halflong)+halflong):] = 0
            #Gets rid of east of Darwin Australia
            interp_mask2[int(round(((90.254/180)*S1))):, (round((98.205/360)*halflong)+halflong):] = 0
            #Gets rid of east of Lam Kaen Thailand
            interp_mask2[int(round(((85.204/180)*S1))):, (round((103.204/360)*halflong)+halflong):] = 0
            #Gets rid of east of Bintuhan Indonesia
            interp_mask2[int(round(((82.320/180)*S1))):, (round((105.831/360)*halflong)+halflong):] = 0
            #Gets rid of north east of fiddly Indonesia bit 1
            interp_mask2[int(round(((79.686/180)*S1))):, (round((120.452/360)*halflong)+halflong):] = 0
            #Gets rid of north east of fiddly Indonesia bit 2
            if optB == 2: #North Indian remaining
                interp_mask2[:int(round(((90.000/180)*S1))), :] = 0
            if optB == 3: #South Indian remaining
                interp_mask2[int(round(((90.000/180)*S1))):, :] = 0
        elif optA == 3: #Pacific 
            interp_mask2[int(round(((156.563/180)*S1))):, :] = 0
            #Gets rid of north of Arctic circle
            interp_mask2[:int(round(((30.000/180)*S1))), :] = 0
            #Gets rid of Southern Ocean
            interp_mask2[:, (round((289.946/360)*halflong)+halflong):] = 0
            #Gets rid of east of inland Calate Chilea
            interp_mask2[:, :(round((98.885/360)*halflong)+halflong)] = 0
            #Gets rid of west of Kaw Thaung Thailand
            interp_mask2[int(round(((106.35/180)*S1))):, (round((262.1/360)*halflong)+halflong):] = 0
            #Gets rid of north east of Pinotepa national Mexico
            interp_mask2[int(round(((99.314/180)*S1))):, (round((276.111/360)*halflong)+halflong):] = 0
            #Gets rid of north east of savegre Costa Rica
            interp_mask2[:int(round(((72.277/180)*S1))), :(round((143.511/360)*halflong)+halflong)] = 0
            #Gets rid of south west of Tasmania Australia
            interp_mask2[:int(round(((81.116/180)*S1))), :(round((130.247/360)*halflong)+halflong)] = 0
            #Gets rid of south west of Darwin Australia
            interp_mask2[:int(round(((95.032/180)*S1))), :(round((103.315/360)*halflong)+halflong)] = 0
            #Gets rid of south west of Kuantan Malasia
            interp_mask2[:int(round(((82.944/180)*S1))), :(round((112.569/360)*halflong)+halflong)] = 0
            #Gets rid of south west of Gresik Regency Indonesia
            interp_mask2[:int(round(((86.334/180)*S1))), :(round((105.813/360)*halflong)+halflong)] = 0
            #Gets rid of south west of fiddly bit Indonesia 1
            interp_mask2[:int(round(((83.643/180)*S1))), :(round((108.347/360)*halflong)+halflong)] = 0
            #Gets rid of south west of fiddly bit Indonesia 2
            interp_mask2[:int(round(((99.170/180)*S1))), :(round((99.861/360)*halflong)+halflong)] = 0
            #Gets rid of south west of fiddly bit Indonesia 3
            if optB == 2: #North Pacific remaining
                interp_mask2[:int(round(((90.000/180)*S1))), :] = 0
            if optB == 3: #South Pacific remaining
                interp_mask2[int(round(((90.000/180)*S1))):, :] = 0
        elif optA == 4: #Southern Ocean
            interp_mask2[int(round(((30.000/180)*S1))):, :] = 0
            #Gets rid of north of Southern Ocean
            interp_mask2[:, :halflong] = 0
            #Gets rid of west side
        elif optA == 5: #Arctic Ocean
            interp_mask2[:int(round(((156.563/180)*S1))), :] = 0
            #Gets rid of south of Arctic circle
            interp_mask2[:, :halflong] = 0
            #Gets rid of west side
    
        interp_mask = np.copy(interp_mask2)
        if optA == 1: #Atlantic
            interp_mask = np.roll(interp_mask, int(round(halflong/2)), axis=1)
        interp_mask = interp_mask[:, halflong:]
        if optA == 1: #Atlantic
            interp_mask = np.roll(interp_mask, -int(round(halflong/2)), axis=1)   
    
    else:
        interp_mask = np.full((interp_mask.shape[0],interp_mask.shape[1]),1)
    #interp_mask4 = np.logical_not(interp_mask3) #now the land is mask out
    interp_mask4 = np.flip(interp_mask, 0) #standard upside down array now
    interp_mask5 = np.repeat(interp_mask4[np.newaxis, :, :], depth, axis=0) #now 3D
    interp_mask6 = ma.array(interp_mask5) #now a masked array
    
    return interp_mask6
#******************************************************************************
def latitudeband_mask(latnorth, latsouth, lats, longs, nlats, nlongs, depth):
    #latitudeband_mask(latnorth, latsouth, lats, longs, 0 , 0)
    #does not calculate it's own cell midpoints
    #latitudeband_mask(latnorth, latsouth, 0, 0, 64 , 128)
    #does calculate it's own cell midpoints of nlat = 64 nlong = 128
    #latnorth = lats north of here are masked out
    #latsouth = lats south of here are masked out
    import numpy.ma as ma

    if nlats != 0:
        lats, longs1 = cellmidpoints(nlats, nlongs)
    if nlongs != 0:
        lats1, longs = cellmidpoints(nlats, nlongs)

    fullmat = np.full((lats.shape[0], longs.shape[0]), 0)

    abc2 = np.argmax(lats<=latnorth)#88.59375
    if abc2 != 0: 
        fullmat[:abc2,:] = 1 #north masked out
    abc3 = np.argmax(lats<latsouth)#-88.59375
    if abc3 != 0:
        fullmat[abc3:,:] = 1 #south masked out

    fullmat2 = np.repeat(fullmat[np.newaxis, :, :], depth, axis=0) #now 3D
    mask = ma.array(fullmat2) #now a mask
    
    return mask
#******************************************************************************
#******************************************************************************
#******************************************************************************
#******************************************************************************
#******************************************************************************
#******************************************************************************
#******************************************************************************
#******************************************************************************
#******************************************************************************
#******************************************************************************
#******************************************************************************
#******************************************************************************
#******************************************************************************
#******************************************************************************
#******************************************************************************
#******************************************************************************
#******************************************************************************
#******************************************************************************
#******************************************************************************
#******************************************************************************
#******************************************************************************
#******************************************************************************
#******************************************************************************
#******************************************************************************
#******************************************************************************
#******************************************************************************
#******************************************************************************
#******************************************************************************
#******************************************************************************
#******************************************************************************
#******************************************************************************
#******************************************************************************
#******************************************************************************
#******************************************************************************
#******************************************************************************
#******************************************************************************
#******************************************************************************
#******************************************************************************
#******************************************************************************
#******************************************************************************
#******************************************************************************
#******************************************************************************
#******************************************************************************
#******************************************************************************
#******************************************************************************
#******************************************************************************
#******************************************************************************
#******************************************************************************
#******************************************************************************
#******************************************************************************
#******************************************************************************
#******************************************************************************