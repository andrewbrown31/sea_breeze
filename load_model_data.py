import intake
import xarray as xr
import numpy as np
import pandas as pd
import datetime as dt
import metpy.calc as mpcalc
from skimage.segmentation import find_boundaries
import xesmf as xe
import dask.array as da
import scipy
from dask.distributed import progress
import pyproj
import warnings
from tqdm.dask import TqdmCallback

def interp_np(x, xp, fp):
    return np.interp(x, xp, fp, right=np.nan)

def interp_model_level_to_z(z_da,var_da,mdl_dim,heights):

    '''
    Linearly interpolate from model level data to geopotential height levels.
    If the requested height is below the lowest model level, data from the lowest model level is returned.
    Note that for ERA5, the lowest model level is within the first few 10s of meters above the surface.
    If the requested height is above the highest model level, then NaNs are returned.

    Input
    z_da: xarray Dataarray of geopotential height (either AGL or above geoid)
    var_da: xarray Dataarray of variable to interpolate
    mdl_dim: name of the model level dimension (e.g. hybrid). NOTE that model levels must be decreasing (so height is increasing)
    heights: numpy array of height levels
    '''

    assert z_da[mdl_dim][0] > z_da[mdl_dim][-1], "Model levels should be decreasing"

    interp_da = xr.apply_ufunc(interp_np,
                heights,
                z_da,
                var_da,
                input_core_dims=[ ["height"], [mdl_dim], [mdl_dim]],
                output_core_dims=[["height"]],
                exclude_dims=set((mdl_dim,)),
                dask="parallelized",
                output_dtypes=var_da.dtype,
                vectorize=True)
    interp_da["height"] = heights
    
    return interp_da

def load_era5_ml_and_interp(t1,t2,lat_slice,lon_slice,
                            upaths=None,vpaths=None,zpaths=None,
                            heights=np.arange(0,4600,100),chunks={"time":"auto","hybrid":-1}):

    """
    ## Load in ERA5 data that was downladed from the Google cloud. That includes u and v wind components as well as geopotential height. Then, interpolate from model levels to height levels.
    I have tried to name the downloaded files systematically using the same monthly notation as in the rt52 project, however the paths can also be manually specified as lists (expecting one file for each variable)

    t1: start time in %Y-%m-%d %H:%M"
    t1: end time in %Y-%m-%d %H:%M"
    lat_slice: a slice to restrict lat domain
    lon_slice: a slice to restrict lon domain    
    upaths: an array of paths for u data. if none use time and look in gb02 dir
    vpaths: an array of paths for v data. if none use time and look in gb02 dir
    zpaths: an array of paths for z data. if none use time and look in gb02 dir    
    heights: to interpolate to (in metres)
    chunks: dict describing the number of chunks. see xr.open_dataset
    """

    #Load ERA5 model level data downloaded from ERA5
    if (upaths is None) | (vpaths is None) | (zpaths is None):
        time_starts = pd.date_range(
            pd.to_datetime(t1).replace(day=1),t2,freq="MS").strftime("%Y%m%d").astype(int).values
        time_ends = [(t + dt.timedelta(days=32)).replace(day=1) - dt.timedelta(days=1) 
                     for t in pd.to_datetime(time_starts,format="%Y%m%d")]
        upaths = ["/g/data/gb02/ab4502/era5_model_lvl/era5_mdl_u_" + 
                  str(t1) + "_" + t2.strftime("%Y%m%d") +".nc" for t1,t2 in zip(time_starts,time_ends)]
        vpaths = ["/g/data/gb02/ab4502/era5_model_lvl/era5_mdl_v_" + 
                        str(t1) + "_" + t2.strftime("%Y%m%d") +".nc" for t1,t2 in zip(time_starts,time_ends)]
        zpaths = ["/g/data/gb02/ab4502/era5_model_lvl/era5_mdl_z_" + 
                        str(t1) + "_" + t2.strftime("%Y%m%d") +".nc" for t1,t2 in zip(time_starts,time_ends)]                

    #Load the data
    u = xr.combine_by_coords([load_era5_ml(upath,t1,t2,lat_slice,lon_slice,chunks=chunks) for upath in upaths])
    v = xr.combine_by_coords([load_era5_ml(vpath,t1,t2,lat_slice,lon_slice,chunks=chunks) for vpath in vpaths])
    z = xr.combine_by_coords([load_era5_ml(zpath,t1,t2,lat_slice,lon_slice,chunks=chunks) for zpath in zpaths])
    topo,lsm,_= load_era5_static(lon_slice,lat_slice,t1,t2)
    f = xr.merge((u,v,z))

    #Adjust geopotential to height above surface, using topography data saved on gb02
    g = 9.80665    #https://confluence.ecmwf.int/display/CKB/ERA5%3A+compute+pressure+and+geopotential+on+model+levels%2C+geopotential+height+and+geometric+height#heading-Geopotentialheight
    topo = topo / g
    f["geopotential"] = (f["geopotential"] / g) - topo
    f = f.rename({"geopotential":"geopotential_hgt_agl"})

    #Convert to height levels
    interp_v = interp_model_level_to_z(f["geopotential_hgt_agl"],f["v_component_of_wind"],"hybrid",heights)
    interp_u = interp_model_level_to_z(f["geopotential_hgt_agl"],f["u_component_of_wind"],"hybrid",heights)
    interp_era5 = xr.Dataset({"u":interp_u, "v":interp_v})

    return interp_era5, lsm

def load_era5_variable(vnames,t1,t2,lon_slice,lat_slice,chunks="auto"):

    '''
    Load era5 data using the NCI intake catalog

    Input
    vname: list of names of era5 variables
    t1: start time in %Y-%m-%d %H:%M"
    t1: end time in %Y-%m-%d %H:%M"
    lat_slice: a slice to restrict lat domain
    lon_slice: a slice to restrict lon domain

    Output:
    xarray dataset
    '''

    #Set up times to search within catalog
    data_catalog = get_intake_cat_era5()
    time_starts = pd.date_range(pd.to_datetime(t1).replace(day=1),t2,freq="MS").strftime("%Y%m%d").astype(int).values
    time_ends = [(t + dt.timedelta(days=32)).replace(day=1) - dt.timedelta(days=1) for t in pd.to_datetime(time_starts,format="%Y%m%d")]
    times = [str(t1) + "-" + t2.strftime("%Y%m%d") for t1,t2 in zip(time_starts,time_ends)]

    #Load the data using intake
    out = dict.fromkeys(vnames)
    for vname in vnames:
        ds = data_catalog.search(variable=vname,
                                product="era5-reanalysis",
                                time_range=times).\
                                    to_dask(cdf_kwargs={"chunks":chunks}).\
                                        sel(time=slice(t1,t2))
        ds = ds.isel(latitude=slice(None,None,-1))
        ds["longitude"] = (ds.longitude % 360)
        ds = ds.sortby("longitude")    
        ds = ds.rename({"longitude":"lon","latitude":"lat"}).sel(lon=lon_slice, lat=lat_slice)
        out[vname] = ds
        
    return out

def load_era5_static(lon_slice,lat_slice,t1,t2,chunks="auto"):

    '''
    For ERA5, load static variables using the first time step of the period.
    Also flip the latitude coord and convert -180-180 lons to 0-360 (for consistency with BARRA)
    lat_slice: a slice to restrict lat domain
    lon_slice: a slice to restrict lon domain
    t1: start time in %Y-%m-%d %H:%M"
    t1: end time in %Y-%m-%d %H:%M"

    Returns orography, binary land sea mask, and binary lake mask
    '''

    data_catalog = get_intake_cat_era5()
    time_starts = pd.date_range(pd.to_datetime(t1).replace(day=1),t2,freq="MS").strftime("%Y%m%d").astype(int).values
    time_ends = [(t + dt.timedelta(days=32)).replace(day=1) - dt.timedelta(days=1) for t in pd.to_datetime(time_starts,format="%Y%m%d")]
    times = [str(t1) + "-" + t2.strftime("%Y%m%d") for t1,t2 in zip(time_starts,time_ends)]

    #times = pd.date_range(pd.to_datetime(t1).replace(day=1),t2,freq="MS").strftime("%Y%m").astype(int).values
    orog = data_catalog.search(variable="z",product="era5-reanalysis",time_range=times,levtype="sfc").to_dask(cdf_kwargs={"chunks":chunks})
    orog = orog.isel(latitude=slice(None,None,-1))
    orog["longitude"] = (orog.longitude % 360)
    orog = orog.sortby("longitude")    
    orog = orog.rename({"longitude":"lon","latitude":"lat"}).sel(lon=lon_slice, lat=lat_slice).isel(time=0)

    lsm = data_catalog.search(variable="lsm",product="era5-reanalysis",time_range=times).to_dask(cdf_kwargs={"chunks":chunks})
    lsm = lsm.isel(latitude=slice(None,None,-1))
    lsm["longitude"] = (lsm.longitude % 360)
    lsm = lsm.sortby("longitude")
    lsm = lsm.rename({"longitude":"lon","latitude":"lat"}).sel(lon=lon_slice, lat=lat_slice).isel(time=0)

    cl = data_catalog.search(variable="cl",product="era5-reanalysis",time_range=times).to_dask(cdf_kwargs={"chunks":chunks})
    cl = cl.isel(latitude=slice(None,None,-1))
    cl["longitude"] = (cl.longitude % 360)
    cl = cl.sortby("longitude")
    cl = cl.rename({"longitude":"lon","latitude":"lat"}).sel(lon=lon_slice, lat=lat_slice).isel(time=0)

    return orog.z, (lsm.lsm >= 0.5) * 1, (cl.cl >= 0.5)

def remove_era5_inland_lakes(lsm,cl):

    '''
    Use the ERA5 lake cover mask (cl) to assign inland lakes as land points in the land sea mask (lsm)
    '''
    return xr.where(cl,1,lsm)
    #return xr.where((lsm.lon>135) & (lsm.lon<142) & (lsm.lat>-32.25) & (lsm.lat<-25), 1, lsm).T

def load_era5_ml(path,t1,t2,lat_slice,lon_slice,chunks={"time":"auto","hybrid":-1}):

    '''
    Load ERA5 model level data, as downloaded from the Google cloud and stored on gb02 (using era5_download_google.ipynb)
    t1: start time in %Y-%m-%d %H:%M"
    t1: end time in %Y-%m-%d %H:%M"
    lat_slice: a slice to restrict lat domain
    lon_slice: a slice to restrict lon domain    
    '''

    if ("lat" in chunks.keys()) | ("lon" in chunks.keys()):
        print("WARNING: LAT OR LON IN CHUNKS IS BEING IGNORED, SHOULD BE LATITUDE OR LONGITUDE...")

    #Load data from disk
    f = xr.open_dataset(path,chunks=chunks)
    f = f.rename({"longitude":"lon","latitude":"lat"}).sel(time=slice(t1,t2))
    
    #Reverse model level data for interpolation, as well as the lat dimension, and slice
    f = f.isel(hybrid=slice(None,None,-1))
    f = f.isel(lat=slice(None,None,-1))
    f = f.sel(lat=lat_slice,lon=lon_slice)

    return f

def era5_sfc_moisture(era5_vars):

    """
    From a dict of ERA5 variables, calculate specific humidity and thetae
    Assumes era5_vars contains "sp", "2d", and "2t"
    """

    era5_vars["q"] = mpcalc.mixing_ratio_from_specific_humidity(
        mpcalc.specific_humidity_from_dewpoint(era5_vars["sp"]["sp"],era5_vars["2d"]["d2m"]))
    era5_vars["thetae"] = mpcalc.equivalent_potential_temperature(
        era5_vars["sp"]["sp"], era5_vars["2t"]["t2m"], era5_vars["2d"]["d2m"])
    
    return era5_vars

def get_intake_cat_barra():

    '''
    Return the intake catalog for barra
    '''

    #See here: https://opus.nci.org.au/pages/viewpage.action?pageId=264241965
    data_catalog = intake.open_esm_datastore("/g/data/ob53/catalog/v2/esm/catalog.json")

    return data_catalog

def get_intake_cat_era5():

    '''
    Return the intake catalog for era5
    '''

    #See here: https://opus.nci.org.au/pages/viewpage.action?pageId=264241965
    data_catalog = intake.open_esm_datastore("/g/data/rt52/catalog/v2/esm/catalog.json")

    return data_catalog

def load_barra_variable(vname, t1, t2, domain_id, freq, lat_slice, lon_slice, chunks="auto"):

    '''
    vnames: name of barra variables
    t1: start time in %Y-%m-%d %H:%M"
    t2: end time in %Y-%m-%d %H:%M"
    domain_id: for barra, either AUS-04 or AUST-11
    freq: frequency string (e.g. 1h)
    lat_slice: a slice to restrict lat domain
    lon_slice: a slice to restrict lon domain
    chunks: dict describing the number of chunks. see xr.open_dataset
    '''

    data_catalog = get_intake_cat_barra()
    times = pd.date_range(pd.to_datetime(t1).replace(day=1),t2,freq="MS").strftime("%Y%m").astype(int).values
    out = []
    da = data_catalog.search(
        variable_id=vname,
        domain_id=domain_id,
        freq=freq,
        start_time=times)\
            .to_dask(cdf_kwargs={"chunks":chunks}).\
                sel(lon=lon_slice, lat=lat_slice, time=slice(t1,t2))[vname]
        
    return da

def load_barra_static(domain_id,lon_slice,lat_slice):

    '''
    For a barra domain, load static variables
    domain_id: for barra, either AUS-04 or AUST-11
    lat_slice: a slice to restrict lat domain
    lon_slice: a slice to restrict lon domain
    '''

    data_catalog = get_intake_cat_barra()
    orog = data_catalog.search(variable_id="orog",domain_id=domain_id).to_dask().sel(lon=lon_slice, lat=lat_slice)
    lsm = data_catalog.search(variable_id="sftlf",domain_id=domain_id).to_dask().sel(lon=lon_slice, lat=lat_slice)

    return orog.orog, (lsm.sftlf >= 0.5) * 1

def barra_sfc_moisture(huss,ps,tas):

    """
    From a dict of BARRA variables, calculate specific humidity, dewpoint, and theta-e
    Assumes barra_vars contains "huss", "ps", and "tas"
    """

    q = mpcalc.mixing_ratio_from_specific_humidity(huss)
    dp = mpcalc.dewpoint_from_specific_humidity(ps, tas, huss)
    thetae = mpcalc.equivalent_potential_temperature(ps, tas, dp)

    return q, dp, thetae

def load_aus2200_static(exp_id,lon_slice,lat_slice,chunks="auto"):

    '''
    Load static fields for the mjo-enso AUS2200 experiment, stored on the ua8 project

    ## Input
    * exp_id: string describing the experiment. either 'mjo-elnino', 'mjo-lanina' or 'mjo-neutral'

    * lat_slice: a slice to restrict lat domain

    * lon_slice: a slice to restrict lon domain
    '''

    assert exp_id in ['mjo-elnino', 'mjo-lanina', 'mjo-neutral'], "exp_id must either be 'mjo-elnino', 'mjo-lanina' or 'mjo-neutral'"
    
    orog = xr.open_dataset("/g/data/ua8/AUS2200/"+exp_id+"/v1-0/fx/orog/orog_AUS2200_"+exp_id+"_fx.nc",chunks=chunks).\
            sel(lat=lat_slice,lon=lon_slice)
    lsm = xr.open_dataset("/g/data/ua8/AUS2200/"+exp_id+"/v1-0/fx/lmask/lmask_AUS2200_"+exp_id+"_fx.nc",chunks=chunks).\
            sel(lat=lat_slice,lon=lon_slice)

    return orog.orog, ((lsm.lmask==100)*1)

def load_aus2200_variable(vname, t1, t2, exp_id, lon_slice, lat_slice, freq, hgt_slice=None, chunks="auto", staggered=None, dx=0.022):

    '''
    Load variables from the mjo-enso AUS2200 experiment, stored on the ua8 project

    ## Input

    * vnames: names of aus2200 variable to load

    * t1: start time in %Y-%m-%d %H:%M"

    * t2: start time in %Y-%m-%d %H:%M"
    
    * exp_id: string describing the experiment. either 'mjo-elnino', 'mjo-lanina' or 'mjo-neutral'

    * lat_slice: a slice to restrict lat domain

    * lon_slice: a slice to restrict lon domain

    * freq: time frequency (string). either "10min", "1hr", "1hrPlev"

    * hgt_slice: a slice to restrict data in the vertical (in m)

    * chunks: dict describing the number of chunks. see xr.open_dataset
    '''

    #This code makes sure the inputs for experiment id and time frequency match what is on disk 
    assert exp_id in ['mjo-elnino', 'mjo-lanina', 'mjo-neutral'], "exp_id must either be 'mjo-elnino', 'mjo-lanina' or 'mjo-neutral'"
    assert freq in ["10min", "1hr", "1hrPlev"], "exp_id must either be '10min', '1hr', '1hrPlev'"

    #We are loading a list of files from disk using xr.open_mfdataset. This preprocessing 
    # just slices the lats, lons and levels we are interested in for each file, which is more efficient
    def _preprocess(ds):
        ds = ds.sel(lat=lat_slice,lon=lon_slice)
        return ds
    def _preprocess_hgt(ds):
            ds = ds.sel(lat=lat_slice,lon=lon_slice,lev=hgt_slice)
            return ds   

    if staggered is not None:
        _, lsm = load_aus2200_static(exp_id,lon_slice,lat_slice)
        if staggered == "lat":
            lat_slice=slice(lat_slice.start-(dx*0.5),lat_slice.stop+(dx*0.5))
        elif staggered == "lon":
            lon_slice=slice(lon_slice.start-(dx*0.5),lon_slice.stop+(dx*0.5))
        elif staggered == "time":
            if freq == "10min":
                time_delta = dt.timedelta(minutes=10)
                freq_str = freq
            elif (freq == "1hr") | (freq == "1hrPlev"):
                time_delta = dt.timedelta(hours=1)
                freq_str = "1h"
            unstaggered_times = pd.date_range(t1,t2,freq=freq_str)
            t1 = pd.to_datetime(t1) - time_delta
            t2 = pd.to_datetime(t2) + time_delta
        else:
            raise ValueError("Invalid stagger dim")
    
    fnames = "/g/data/ua8/AUS2200/"+exp_id+"/v1-0/"+freq+"/"+vname+"/"+vname+"_AUS2200_"+exp_id+"_*.nc"
    if hgt_slice is not None:
        da = xr.open_mfdataset(fnames, 
                               chunks=chunks, 
                               parallel=True, 
                               preprocess=_preprocess_hgt).sel(time=slice(t1,t2))[vname]
    else:
        da = xr.open_mfdataset(fnames,
                               chunks=chunks,
                               parallel=True,
                               preprocess=_preprocess).sel(time=slice(t1,t2))[vname]
    
    if staggered == "lat":
        da = (da.isel(lat=slice(0,-1)).assign_coords({"lat":lsm.lat}) +
                        da.isel(lat=slice(1,da.lat.shape[0])).assign_coords({"lat":lsm.lat})) / 2        
    if staggered == "lon":
        da = (da.isel(lon=slice(0,-1)).assign_coords({"lon":lsm.lon}) +
                       da.isel(lon=slice(1,da.lon.shape[0])).assign_coords({"lon":lsm.lon})) / 2         
    if staggered == "time":
        da = (da.isel(time=slice(0,-1)).assign_coords({"time":unstaggered_times}) +\
                    da.isel(time=slice(1,da.time.shape[0])).assign_coords({"time":unstaggered_times})) / 2
    
    return da

def round_times(ds,freq):
    
    """
    For dataarray, round the time coordinate to the nearst freq

    Example: aus2200 time values are sometimes very slightly displaced from a 10-minute time step
    """

    #Round the time stamps so that they are easier to work with
    if freq in ["1hrPlev","1hr"]:
        ds["time"] = ds.time.dt.round("1h")
    elif freq in ["10min"]:
        ds["time"] = ds.time.dt.round("10min")
    else:
        raise Exception("Not sure of the time frequency to round to")

    return ds

def interp_times(da,interp_times,method="linear",lower_bound=None):

    """
    For a dataarray, interpolate in time.
    If a lower bound is given, then extrapolation is not allowed below this
    """
        
    da = da.interp(coords={"time":interp_times},method=method,kwargs={"fill_value":"extrapolate"})
    if lower_bound is not None:
        da = xr.where(da < lower_bound, lower_bound, da)
        
    return da

def destagger_aus2200(ds_dict,destag_list,interp_to=None,lsm=None):

    """
    
    From a dictionary of aus2200 datasets (output from load_aus2200_variable), destagger variables in destag_list by interpolating
    
    ## Input
    * ds_dict: a dictionary of aus2200 xarray datasets. output from load_aus2200_variable()

    * destag_list: list of variables to destagger

    * interp_to: variable for which to use spatial info to interp onto

    * lsm: land sea mask dataset to interp on to

    ## Output
    a dictionary of datasets with destaggered variables

    ## Example
    destagger_aus2200(ds_dict, ["uas","vas"], "hus")

    """

    for vars in destag_list:
        if interp_to is not None:
            ds_dict[vars] = ds_dict[vars].interp({"lon":ds_dict[interp_to].lon,"lat":ds_dict[interp_to].lat},method="linear")
        elif lsm is not None:
            ds_dict[vars] = ds_dict[vars].interp({"lon":lsm.lon,"lat":lsm.lat},method="linear")
        else:
            raise Exception("Need to input either a variable to interp to, or a land sea mask, to get spatial info")
        
    return ds_dict

def get_weights(x, p=4, q=2, R=5, slope=-1, r=10000):
    """
    Calculate weights for averaging angles between pixels and coastlines
    
    Method:
    x the distance
    Let y1 = m1 * (x / R) ** (-p) for x > R.
    Let y2 = S - m2 * (x / R) ** (q) for x <= R.
    Equate y1 and y2 and their derivative at x = R to get
    S = m1 + m2
    slope = -p * m1 = -q * m2 => m1 = -slope/p and m2 = -slope/q
    Thus specifying p, q, R, and the function's slope at x=R determines m1, m2 and S.

    # Inputs

    * x: Distance (array like)

    * p: Inverse power to decrease weights after distance R (float)

    * q: Inverse power to decrease weights before distance R (float)

    * R: Distance (in x) to change inverse weighting power from p to q

    * slope: Slpe of function at point R

    * r: The distance at which the weights go to zero (to avoid overflows)

    From Ewan Short
    """
    m1 = -slope/p
    m2 = -slope/q
    S = m1 + m2
    y = da.where(x>R,  m1 * (x / R) ** (-p), S - m2 * (x / R) ** (q))
    y = da.where(x==0, np.nan, y)
    y = da.where(x>r, 0, y)
    return y

def get_coastline_angle_kernel(lsm=None,R=20,latlon_chunk_size=10,compute=True,path_to_load=None,save=False,path_to_save=None,lat_slice=None,lon_slice=None):

    '''
    Ewan's method with help from Jarrah.
    
    Construct a "kernel" for each coastline point based on the angle between that point and all other points in the domain, then take a weighted average. The weighting function can be customised, but is by default an inverse parabola to distance R, then decreases by distance**4. The weights are set to zero at a distance of 2000 km, and are undefined at the coast (where linear interpolation is done to fill in the coastline gaps)

    ## Input
    * lsm: xarray dataarray with a binary lsm, and lat lon info

    * R: the distance at which the weighting function is changed from 1/p to 1/q (in km). See get_weights function

    * coast_dim_chunk_size: the size of the chunks over the coastline dimension

    * extend_lon: only valid for global data that is periodic in longitude (with lons ranging from -180 to 180). How many pixels to extend the east and west boundary? Noting that pixels at the extreme east/west of the domain could be impacted by coastlines on the opposite boundary. The value given (int) will extend the E/W boundary by as many pixels.

    * compute: boolean whether or not to actually compute the angles, or just load from disk (in which case path_to_load must be specified)

    * path_to_load: file path to previous output that can be loaded

    * save: boolean - save the angles output?

    * path_to_save: file path to save output

    * lat_slice: if not computing, lats to slice when loading angles from disk

    * lat_slice: if not computing, lons to slice when loading angles from disk

    ## Output
    * An xarray dataset with an array of coastline angles (0-360 degrees from N) for the labelled coastline array, as well as an array of angle variance as an estimate of how many coastlines are influencing a given point
    '''

    if save:
        if path_to_save is None:
            raise AttributeError("Saving but no path speficfied")
        
    if compute:

        assert np.in1d([0,1],np.unique(lsm)).all(), "Land-sea mask must be binary"
        
        warnings.simplefilter("ignore")

        #From the land sea mask define the coastline and a label array
        coast_label = find_boundaries(lsm)*1
        land_label = lsm.values

        #Get lat lon info for domain and coastline, and convert to lower precision
        lon = lsm.lon.values
        lat = lsm.lat.values
        xx,yy = np.meshgrid(lon,lat)
        xx = xx.astype(np.float16)
        yy = yy.astype(np.float16)    

        #Define coastline x,y indices from the coastline mask
        xl, yl = np.where(coast_label)

        #Get coastline lat lon vectors
        yy_t = np.array([yy[xl[t],yl[t]] for t in np.arange(len(yl))])
        xx_t = np.array([xx[xl[t],yl[t]] for t in np.arange(len(xl))])

        #Repeat the 2d lat lon array over a third dimension (corresponding to the coast dim). Also repeat the yy_t and xx_t vectors over the spatial arrays
        yy_rep = da.moveaxis(da.stack([da.from_array(yy)]*yl.shape[0],axis=0),0,-1).rechunk({0:-1,1:-1,2:latlon_chunk_size})
        xx_rep = da.moveaxis(da.stack([da.from_array(xx)]*xl.shape[0],axis=0),0,-1).rechunk({0:-1,1:-1,2:latlon_chunk_size})
        xx_t_rep = (xx_rep * 0) + xx_t
        yy_t_rep = (yy_rep * 0) + yy_t

        #Calculate the distance and angle between coastal points and all other points using pyproj, then convert to complex space.
        geod = pyproj.Geod(ellps="WGS84")
        def calc_dist(lon1,lat1,lon2,lat2):
            fa,_,d = geod.inv(lon1,lat1,lon2,lat2)
            return d/1e3 * np.exp(1j * np.deg2rad(fa))
        
        stack = da.map_blocks(
                    calc_dist,
                    xx_t_rep,
                    yy_t_rep,
                    xx_rep,
                    yy_rep,
                    dtype=np.complex64,
                    meta=np.array((), dtype=np.complex64))
        del xx_t_rep, yy_t_rep, yy_rep, xx_rep
        
        #Move axes around for convenience later
        stack = da.moveaxis(stack, -1, 0)

        #Get back distance by taking absolute value
        stack_abs = da.abs(stack,dtype=np.float32)
        
        #Create an inverse distance weighting function
        weights = get_weights(stack_abs, p=4, q=2, R=R, slope=-1)

        #Take the weighted mean and convert complex numbers to an angle and magnitude
        print("INFO: Take the weighted mean and convert complex numbers to an angle and magnitude...")
        mean_angles = da.mean((weights*stack), axis=0).persist()
        progress(mean_angles)
        mean_abs = da.abs(mean_angles)
        mean_angles = da.angle(mean_angles)    

        #Flip the angles inside the coastline for convention, and convert range to 0 to 2*pi
        mean_angles = da.where(land_label==1,(mean_angles+np.pi) % (2*np.pi),mean_angles % (2*np.pi))

        #Calculate the weighted circular variance
        print("INFO: Calculating the sum of the weights...")
        total_weight = da.sum(weights, axis=0).persist()
        progress(total_weight)
        print("INFO: Calculating variance...")
        variance = (1 - da.abs(da.sum( (weights/total_weight) * (stack / stack_abs), axis=0))).persist()
        progress(variance)
        del stack, weights, total_weight 

        #Calculate minimum distance to the coast
        print("INFO: Calculating minimum distance to the coast...")
        min_coast_dist = stack_abs.min(axis=0).persist()

        #Convert angles to degrees, and from bearing to orientation of coastline.
        #Also create an xr dataarray object
        angle_da = xr.DataArray(da.rad2deg(mean_angles) - 90,coords={"lat":lat,"lon":lon})
        angle_da = xr.where(angle_da < 0, angle_da+360, angle_da)      

        #Convert variance and coast arrays to xr dataarrays
        var_da = xr.DataArray(variance,coords={"lat":lat,"lon":lon})
        coast = xr.DataArray(coast_label,coords={"lat":lat,"lon":lon})
        mean_abs = xr.DataArray(mean_abs,coords={"lat":lat,"lon":lon})
        mean_angles = xr.DataArray(mean_angles,coords={"lat":lat,"lon":lon})
        min_coast_dist = xr.DataArray(min_coast_dist,coords={"lat":lat,"lon":lon})

        #Create an xarray dataset
        angle_ds =  xr.Dataset({
            "angle":angle_da,
            "variance":var_da,
            "coast":coast,
            "mean_abs":mean_abs,
            "mean_angles":mean_angles,
            "min_coast_dist":min_coast_dist})

        #Do the interpolation across the coastline
        angle_ds = interpolate_angles(angle_ds)

    else:

        if path_to_load is not None:
            angle_ds = xr.open_dataset(path_to_load)
            if lat_slice is not None:
                angle_ds = angle_ds.sel(lat=lat_slice)
            if lon_slice is not None:
                angle_ds = angle_ds.sel(lon=lon_slice)
            save = False
        else:
            raise AttributeError("If not computing the angles, path_to_load needs to be specified")

    if save:
        angle_ds.to_netcdf(path_to_save)

    return angle_ds

def interpolate_angles(angle_ds):

    """
    From a dataset of coastline angles, interpolate across the coastline.

    This is used because the result of get_coastline_angle_kernel() is not defined along the coastline.
    """

    xx,yy = np.meshgrid(angle_ds.lon,angle_ds.lat)

    mean_complex = angle_ds.mean_abs * da.exp(1j*angle_ds.mean_angles)
    points = mean_complex.values.ravel()
    valid = ~np.isnan(points)
    points_valid = points[valid]
    xx_rav, yy_rav = xx.ravel(), yy.ravel()
    xxv = xx_rav[valid]
    yyv = yy_rav[valid]
    interpolated_angles = scipy.interpolate.griddata(np.stack([xxv, yyv]).T, points_valid, (xx, yy), method="linear").reshape(xx.shape) 

    interpolated_angles = da.rad2deg(da.angle(interpolated_angles))
    interpolated_angle_da = xr.DataArray(interpolated_angles - 90,coords={"lat":angle_ds.lat,"lon":angle_ds.lon})
    interpolated_angle_da = xr.where(interpolated_angle_da < 0, interpolated_angle_da+360, interpolated_angle_da)  

    angle_ds = angle_ds.drop_vars(["mean_abs","mean_angles"])
    angle_ds["angle_interp"] = interpolated_angle_da

    return angle_ds
