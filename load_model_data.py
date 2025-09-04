import intake
import xarray as xr
import numpy as np
import pandas as pd
import datetime as dt
import metpy.calc as mpcalc
from skimage.segmentation import find_boundaries
#import xesmf as xe
import dask.array as da
import scipy
from dask.distributed import progress
import pyproj
import warnings
import glob

def interp_scipy(x, xp, fp):
    f = scipy.interpolate.interp1d(xp, fp, kind="linear", fill_value="extrapolate")
    return f(x)

def interp_model_level_to_z(z_da,var_da,mdl_dim,heights,model="ERA5"):

    """
    Linearly interpolate from model level data to height levels. Supported for AUS2200 and ERA5 data.

    Parameters
    ----------
    z_da : xarray.DataArray
        Height data (either AGL or above geoid) for each model level.
    var_da : xarray.DataArray
        Variable to interpolate, with the same model level dimension as z_da.
    mdl_dim : str
        Name of the model level dimension (e.g., 'lev', 'hybrid'). Model levels must be decreasing (height increasing).
    heights : numpy.ndarray
        Array of target height levels to interpolate to.
    model : str, optional
        Model name, either "ERA5" or "AUS2200"

    Returns
    -------
    xarray.DataArray
        Variable interpolated to the specified height levels. If the requested height is below the lowest model level, data from the lowest model level is returned. If above the highest model level, NaNs are returned.

    Notes
    -----
    - If the requested height is below the lowest model level, data from the lowest model level is returned.
    - Note that for ERA5, the lowest model level is within the first few 10s of meters above the surface.
    - If the requested height is above the highest model level, then NaNs are returned.
    """

    if model=="ERA5":
        assert z_da[mdl_dim][0] > z_da[mdl_dim][-1], "Model levels should be decreasing"

    interp_da = xr.apply_ufunc(interp_scipy,
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
    upaths: an array of paths for u data. if none use time and look in ng72 dir
    vpaths: an array of paths for v data. if none use time and look in ng72 dir
    zpaths: an array of paths for z data. if none use time and look in ng72 dir    
    heights: to interpolate to (in metres)
    chunks: dict describing the number of chunks. see xr.open_dataset
    """

    #Load ERA5 model level data downloaded from ERA5
    if (upaths is None) | (vpaths is None) | (zpaths is None):
        time_starts = pd.date_range(
            pd.to_datetime(t1).replace(day=1),t2,freq="MS").strftime("%Y%m%d").astype(int).values
        time_ends = [(t + dt.timedelta(days=32)).replace(day=1) - dt.timedelta(days=1) 
                     for t in pd.to_datetime(time_starts,format="%Y%m%d")]
        upaths = ["/g/data/ng72/ab4502/era5_model_lvl/era5_mdl_u_component_of_wind_" + 
                  str(t1) + "_" + t2.strftime("%Y%m%d") +".nc" for t1,t2 in zip(time_starts,time_ends)]
        vpaths = ["/g/data/ng72/ab4502/era5_model_lvl/era5_mdl_v_component_of_wind_" + 
                        str(t1) + "_" + t2.strftime("%Y%m%d") +".nc" for t1,t2 in zip(time_starts,time_ends)]
        zpaths = ["/g/data/ng72/ab4502/era5_model_lvl/era5_mdl_geopotential_" + 
                        str(t1) + "_" + t2.strftime("%Y%m%d") +".nc" for t1,t2 in zip(time_starts,time_ends)]                

    #Load the data
    u = xr.combine_by_coords([load_era5_ml(upath,t1,t2,lat_slice,lon_slice,chunks=chunks) for upath in upaths])
    v = xr.combine_by_coords([load_era5_ml(vpath,t1,t2,lat_slice,lon_slice,chunks=chunks) for vpath in vpaths])
    z = xr.combine_by_coords([load_era5_ml(zpath,t1,t2,lat_slice,lon_slice,chunks=chunks) for zpath in zpaths])
    topo,lsm,_= load_era5_static(lon_slice,lat_slice,t1,t2)
    f = xr.merge((u,v,z))

    #Adjust geopotential to height above surface, using topography data saved on ng72
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
    Load ERA5 model level data, as downloaded from the Google cloud and stored on ng72 (using era5_download_google.ipynb)
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

def load_barra_variable(vname, t1, t2, domain_id, freq, lat_slice, lon_slice, chunks="auto", smooth=False, sigma=2, smooth_axes=None):

    '''
    vnames: name of barra variables
    t1: start time in %Y-%m-%d %H:%M"
    t2: end time in %Y-%m-%d %H:%M"
    domain_id: for barra, either AUS-04 or AUST-11
    freq: frequency string (e.g. 1h)
    lat_slice: a slice to restrict lat domain
    lon_slice: a slice to restrict lon domain
    chunks: dict describing the number of chunks. see xr.open_dataset
    smooth: boolean - smooth the data using a gaussian filter
    sigma: if smoothing, the sigma of the gaussian filter
    smooth_axes: if smoothing, the axes to smooth over, as an iterable
    '''

    if domain_id in ["AUST-04"]:
        model = "BARRA-C2"
    elif domain_id in ["AUST-11","AUS-11"]:
        model = "BARRA-R2"
    else:
        raise ValueError("Invalid domain id")

    #data_catalog = get_intake_cat_barra()
    times = pd.date_range(pd.to_datetime(t1).replace(day=1),t2,freq="MS").strftime("%Y%m").astype(int).values
    files = [glob.glob("/g/data/ob53/BARRA2/output/reanalysis/"\
                    +domain_id+"/BOM/ERA5/historical/hres/"+model+\
                        "/v1/"+freq+"/"+vname+"/latest/"+\
                            vname+"_"+domain_id+"_*_"+str(t)+"-*.nc") for t in times]
    da = xr.open_mfdataset(
        np.concatenate(files),
        chunks=chunks).\
                sel(lon=lon_slice, lat=lat_slice, time=slice(t1,t2))[vname]
    #da = data_catalog.search(
    #    variable_id=vname,
    #    domain_id=domain_id,
    #    freq=freq,
    #    start_time=times)\
    #        .to_dask(cdf_kwargs={"chunks":chunks}).\
    #            sel(lon=lon_slice, lat=lat_slice, time=slice(t1,t2))[vname]
    
    #Optional smoothing
    da = da.assign_attrs({"smoothed":smooth})
    if smooth:

        if smooth_axes is not None:
            for ax in smooth_axes:
                chunks[ax] = -1
            smooth_axes = (np.where(np.in1d(da.isel(time=0).dims,smooth_axes))[0])
        else:
            chunks["lon"] = -1
            chunks["lat"] = -1

        da = da.map_blocks(
            gaussian_filter_time_slice,
            kwargs={"sigma":sigma,"axes":smooth_axes},
            template=da
        )
        da = da.assign_attrs({"gaussian_smoothing_sigma":sigma})
        
    return da

def load_barra_static(domain_id,lon_slice,lat_slice):

    '''
    For a barra domain, load static variables
    domain_id: for barra, either AUS-04 or AUST-11
    lat_slice: a slice to restrict lat domain
    lon_slice: a slice to restrict lon domain
    '''

    # data_catalog = get_intake_cat_barra()
    # orog = data_catalog.search(variable_id="orog",domain_id=domain_id).to_dask().sel(lon=lon_slice, lat=lat_slice)
    # lsm = data_catalog.search(variable_id="sftlf",domain_id=domain_id).to_dask().sel(lon=lon_slice, lat=lat_slice)

    if domain_id in ["AUST-04"]:
        model = "BARRA-C2"
    elif domain_id in ["AUST-11","AUS-11"]:
        model = "BARRA-R2"
    else:
        raise ValueError("Invalid domain id")

    orog_path = "/g/data/ob53/BARRA2/output/reanalysis/"\
                    +domain_id+"/BOM/ERA5/historical/hres/"+model+\
                        "/v1/fx/orog/latest/"+\
                            "orog_*.nc"
    lsm_path = "/g/data/ob53/BARRA2/output/reanalysis/"\
                    +domain_id+"/BOM/ERA5/historical/hres/"+model+\
                        "/v1/fx/sftlf/latest/"+\
                            "sftlf*.nc"
    orog = xr.open_mfdataset(orog_path).sel(lon=lon_slice, lat=lat_slice)
    lsm = xr.open_mfdataset(lsm_path).sel(lon=lon_slice, lat=lat_slice)

    return orog.orog, (lsm.sftlf >= 50) * 1

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

    """
    Load static fields for the mjo-enso AUS2200 experiment, stored on the bs94 project.

    Parameters
    ----------
    exp_id : str
        Experiment ID. Must be one of 'mjo-elnino2016', 'mjo-lanina2018', or 'mjo-neutral2013'.
    lon_slice : slice or array-like
        Slice or indices to restrict longitude domain.
    lat_slice : slice or array-like
        Slice or indices to restrict latitude domain.
    chunks : str or dict, optional
        Chunking for xarray open_mfdataset (default is "auto").

    Returns
    -------
    orog : xarray.DataArray
        Orography field for the selected domain.
    lsm : xarray.DataArray
        Binary land-sea mask (1 for land, 0 for sea) for the selected domain.
    """

    assert exp_id in ['mjo-elnino2016', 'mjo-lanina2018', 'mjo-neutral2013'], "exp_id must either be 'mjo-elnino2016', 'mjo-lanina2018' or 'mjo-neutral2013'"
    
    orog = xr.open_mfdataset("/g/data/bs94/AUS2200/"+exp_id+"/v1-0/fx/orog/orog_AUS2200_*_fx.nc",chunks=chunks).\
            sel(lat=lat_slice,lon=lon_slice)
    lsm = xr.open_mfdataset("/g/data/bs94/AUS2200/"+exp_id+"/v1-0/fx/lmask/lmask_AUS2200_*_fx.nc",chunks=chunks).\
            sel(lat=lat_slice,lon=lon_slice)

    return orog.orog, ((lsm.lmask==100)*1)

def gaussian_filter_time_slice(time_slice,sigma,axes):
    """
    Apply a gaussian filter to a time slice of data. For use with map_blocks
    """
    #out_ds = xr.DataArray(scipy.ndimage.gaussian_filter(time_slice.squeeze(), sigma, axes=axes),
    #                      dims=time_slice.squeeze().dims, coords=time_slice.squeeze().coords)
    out_ds = xr.DataArray(scipy.ndimage.gaussian_filter(
        time_slice.isel(time=0), sigma, axes=axes
        ),dims=time_slice.isel(time=0).dims, coords=time_slice.isel(time=0).coords)
    out_ds = out_ds.expand_dims("time")
    out_ds["time"] = time_slice.time
    return out_ds

def load_aus2200_variable(vname, t1, t2, exp_id, lon_slice, lat_slice, freq, hgt_slice=None, chunks="auto", staggered=None, dx=0.022, smooth=False, smooth_axes=None, sigma=2, interp_hgts=False, dh=100):

    """
    Load variables from the mjo-enso AUS2200 experiment, stored on the bs94 project.

    Parameters
    ----------
    vname : str
        Name of AUS2200 variable to load.
    t1 : str
        Start time in "%Y-%m-%d %H:%M".
    t2 : str
        End time in "%Y-%m-%d %H:%M".
    exp_id : str
        Experiment ID. Must be one of 'mjo-elnino2016', 'mjo-lanina2018', or 'mjo-neutral2013'.
    lon_slice : slice or array-like
        Slice or indices to restrict longitude domain.
    lat_slice : slice or array-like
        Slice or indices to restrict latitude domain.
    freq : str
        Time frequency. Must be "10min", "1hr", or "1hrPlev".
    hgt_slice : slice or array-like, optional
        Slice to restrict data in the vertical (in m).
    chunks : str or dict, optional
        Chunking for xarray open_mfdataset (default is "auto").
    staggered : str, optional
        If not None, the data is staggered in the specified dimension ("lat", "lon", or "time").
    dx : float, optional
        The distance to stagger the data by if staggered in lat or lon (in degrees, default is 0.022).
    smooth : bool, optional
        If True, smooth the data using a Gaussian filter.
    smooth_axes : iterable, optional
        If smoothing, the axes to smooth over.
    sigma : float, optional
        If smoothing, the sigma of the Gaussian filter (default is 2).
    interp_hgts : bool, optional
        If True, interpolate the data to regular height levels.
    dh : int, optional
        If interpolating to height levels, the height increment (in m, default is 100).

    Returns
    -------
    da : xarray.DataArray
        The requested variable, optionally smoothed and/or interpolated to regular height levels.

    Notes
    -----
    - This code currently does not support interpolating in height, as the relevant AUS2200 variable (Z_agl) is not on bs94. A version of Z_agl has been pre-computed and stored on ng72, and this is used if interp_hgts is True, but may not be accessible to all users.
    - If the data is being smoothed or interpolated, the relevant dimensions are set to -1 in the chunks dict.
    - De-staggering in lat/lon can fail at the domain edges depending on the slice used. If this happens, try extending the slice by one grid point in the relevant direction.
    """

    #This code makes sure the inputs for experiment id and time frequency match what is on disk 
    assert exp_id in ['mjo-elnino2016', 'mjo-lanina2018', 'mjo-neutral2013'], "exp_id must either be 'mjo-elnino2016', 'mjo-lanina2018' or 'mjo-neutral2013'"
    assert freq in ["10min", "1hr", "1hrPlev"], "exp_id must either be '10min', '1hr', '1hrPlev'"

    #We are loading a list of files from disk using xr.open_mfdataset. This preprocessing 
    # just slices the lats, lons and levels we are interested in for each file, which is more efficient
    def _preprocess(ds):
        ds = ds.sel(lat=lat_slice,lon=lon_slice)
        return ds
    def _preprocess_hgt(ds):
            ds = ds.sel(lat=lat_slice,lon=lon_slice,lev=hgt_slice)
            return ds   

    #Set up the time and lat/lon slices if the data is staggered
    orog, lsm = load_aus2200_static(exp_id,lon_slice,lat_slice)
    if staggered is not None:
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
    
    #Load the data from disk. If hgt_slice is not None, then we are loading 3D data
    fnames = "/g/data/bs94/AUS2200/"+exp_id+"/v1-0/"+freq+"/"+vname+"/"+vname+"_AUS2200*.nc"
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
    
    #Destagger the data if required
    if staggered == "lat":
        da = (da.isel(lat=slice(0,-1)).assign_coords({"lat":lsm.lat}) +
                        da.isel(lat=slice(1,da.lat.shape[0])).assign_coords({"lat":lsm.lat})) / 2        
    if staggered == "lon":
        da = (da.isel(lon=slice(0,-1)).assign_coords({"lon":lsm.lon}) +
                       da.isel(lon=slice(1,da.lon.shape[0])).assign_coords({"lon":lsm.lon})) / 2         
    if staggered == "time":
        da = (da.isel(time=slice(0,-1)).assign_coords({"time":unstaggered_times}) +\
                    da.isel(time=slice(1,da.time.shape[0])).assign_coords({"time":unstaggered_times})) / 2

    #Optional smoothing using gaussian filter
    if smooth:
        if smooth_axes is not None:
            for ax in smooth_axes:
                chunks[ax] = -1
            smooth_axes = (np.where(np.in1d(da.isel(time=0).dims,smooth_axes))[0])
        else:
            chunks["lev"] = -1
            chunks["lat"] = -1
            chunks["lon"] = -1
        da = da.map_blocks(
            gaussian_filter_time_slice,
            kwargs={"sigma":sigma,"axes":smooth_axes},
            template=da
        )

    #Interpolate to regular height levels
    if interp_hgts:
        chunks["lev"] = -1
        #da = da.chunk({"lev":-1}).interp(lev=np.arange(hgt_slice.start,hgt_slice.stop+dh,dh),method="linear",kwargs={"fill_value":"extrapolate"})
        #Created in aus2200_hybrid_height_calc()
        Z_agl = xr.open_zarr("/g/data/ng72/ab4502/sea_breeze_detection/aus2200_z_agl.zarr/",
                             chunks=chunks).Z_agl.sel(lev=da.lev,lat=da.lat,lon=da.lon)
        da = interp_model_level_to_z(
            Z_agl,
            da.chunk({"lev":-1}),
            "lev",
            np.arange(hgt_slice.start,hgt_slice.stop+dh,dh),
            model="AUS2200"
            )        

    da = da.assign_attrs({"smoothed":smooth})
    if smooth:
        da = da.assign_attrs({"gaussian_smoothing_sigma":sigma})

    return da

def round_times(ds,freq):
    
    """
    For dataarray, round the time coordinate to the nearst freq

    Useful for AUS2200 where time values are sometimes very slightly displaced from a 10-minute time step
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
    Calculate weights for averaging angles between pixels and coastlines.
    This function computes weights based on the distance from a coastline, using a piecewise function with different inverse powers before and after a specified distance `R`. The weights smoothly transition at `R` with a specified slope, and are set to zero beyond a cutoff distance `r`.
    Parameters
    ----------
    x : array_like
        Distance(s) from the coastline.
    p : float, optional
        Inverse power to decrease weights after distance `R`. Default is 4.
    q : float, optional
        Inverse power to decrease weights before distance `R`. Default is 2.
    R : float, optional
        Distance at which the inverse weighting power changes from `p` to `q`. Default is 5.
    slope : float, optional
        Slope of the function at point `R`. Default is -1.
    r : float, optional
        The distance at which the weights go to zero (to avoid overflows). Default is 10000.
    Returns
    -------
    y : array_like
        Calculated weights for each input distance.
    Notes
    -----
    The function is based on a method by Ewan Short. 
    
    Method
    -------
    Continuity and smoothness is ensured at `x = R` by equating the function and its derivative at that point.

    Let y1 = m1 * (x / R) ** (-p) for x > R.
    Let y2 = S - m2 * (x / R) ** (q) for x <= R.
    Equate y1 and y2 and their derivative at x = R to get
    S = m1 + m2
    slope = -p * m1 = -q * m2 => m1 = -slope/p and m2 = -slope/q
    Thus specifying p, q, R, and the function's slope at x=R determines m1, m2 and S.

    """

    m1 = -slope/p
    m2 = -slope/q
    S = m1 + m2
    y = da.where(x>R,  m1 * (x / R) ** (-p), S - m2 * (x / R) ** (q))
    y = da.where(x==0, np.nan, y)
    y = da.where(x>r, 0, y)
    return y

def smooth_angles(angles,sigma):
    """
    Smooth angles using a gaussian filter
    Angles is an xarray dataarray from 0 to 360.
    Sigma is the sigma of the gaussian filter
    """
    z = np.exp(1j * np.deg2rad(angles.values))
    z = np.rad2deg(np.angle(scipy.ndimage.gaussian_filter(z, sigma))) % 360
    return xr.DataArray(z,dims=angles.dims,coords=angles.coords)

def get_coastline_angle_kernel(lsm=None,R=20,latlon_chunk_size=10,compute=True,path_to_load=None,save=False,path_to_save=None,lat_slice=None,lon_slice=None,smooth=False,sigma=4):

    """
    If compute is True, calculate the dominant coastline angle for each point in the domain based on a land-sea mask.

    Otherwise just loads the angles from disk.

    If computing, constructs a "kernel" for each point based on the angle between that point and coastline points, then takes a weighted average. The weighting function can be customised, but is by default an inverse parabola to distance R, then decreases by distance**4. The weights are set to zero at a distance of 10,000 km, and are undefined at the coast (where linear interpolation is done to fill in the coastline gaps).

    Parameters
    ----------
    lsm : xarray.DataArray, optional
        Binary land-sea mask with latitude ("lat") and longitude ("lon") information.
    R : int, default=20
        The distance (in km) at which the weighting function is changed from 1/p to 1/q. Around 2 times the grid spacing of the lsm seems appropriate based on initial tests.
    latlon_chunk_size : int, default=10
        The size of the chunks over the latitude/longitude dimension for computation.
    compute : bool, default=True
        Whether to compute the angles or load from disk.
    path_to_load : str, optional
        File path to previous output that can be loaded if compute is False.
    save : bool, default=False
        Whether to save the computed angles output if compute is True.
    path_to_save : str, optional
        File path to save output if save is True.
    lat_slice : slice or array-like, optional
        Latitude indices or values to slice when loading angles from disk.
    lon_slice : slice or array-like, optional
        Longitude indices or values to slice when loading angles from disk.
    smooth : bool, default=False
        Whether to smooth the interpolated angles output using a Gaussian filter.
    sigma : float, default=4
        Sigma value for the Gaussian filter if smoothing.

    Returns
    -------
    xarray.Dataset
        Dataset containing arrays of coastline angles (0-360 degrees from North), as well as an array of angle variance as an estimate of uncertainty. Includes additional fields for coastline mask and minimum distance to the coast.

    Notes
    -----
    Thank you to Ewan Short and Jarrah Harrison-Lofthouse for help developing this method.
    """

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
        xx = xx.astype(np.float32)
        yy = yy.astype(np.float32)    

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
        angle_ds = interpolate_variance(angle_ds)

        #Attributes
        angle_ds["angle"] = angle_ds["angle"].assign_attrs(
            units = "degrees",
            long_name = "Angle of coastline orientation",
            description = "The angle of dominant coastline orientation in degrees from North. Points with a dominant north-south coastline with ocean to the east will have an angle of 0 degrees. The dominant coastline for each point is determined by the weighted mean of the angles between that point and all coastline points in the domain. The weighting function is an inverse parabola to distance R, then decreases by distance**4. The weights are set to zero at a distance of 2000 km, and are undefined at the coast."
            )
        
        angle_ds["variance"] = angle_ds["variance"].assign_attrs(
            units = "[0,1]",
            long_name = "Variance of coastline angles",
            description = "For each point, the variance of the coastline angles in the domain. This is a measure of how many coastlines are influencing a given point. A value of 0 indicates that coastlines are generally in agreement, and a value of 1 indicates that the point is influenced by coastlines in all directions."
            )
        
        angle_ds["coast"] = angle_ds["coast"].assign_attrs(
            units = "[0,1]",
            long_name = "Coastline mask",
            description = "A binary mask of the coastline determined from the land-sea mask. 1 indicates a coastline point, and 0 indicates a non-coastline point."
            )
        
        angle_ds["min_coast_dist"] = angle_ds["min_coast_dist"].assign_attrs(
            units = "km",
            long_name = "Minimum distance to the coast",
            description = "The minimum distance to the coast for each point in the domain."
            )
        
        angle_ds["angle_interp"] = angle_ds["angle_interp"].assign_attrs(
            units = "degrees",
            long_name = "Interpolated coastline orientation",
            description = "The angle of dominant coastline orientation in degrees from North. Points with a dominant north-south coastline with ocean to the east will have an angle of 0 degrees. The dominant coastline for each point is determined by the weighted mean of the angles between that point and all coastline points in the domain. The weighting function is an inverse parabola to distance R, then decreases by distance**4. The weights are set to zero at a distance of 2000 km, and are undefined at the coast, where linear interpolation is then done."
            )
        
        angle_ds["variance_interp"] = angle_ds["variance_interp"].assign_attrs(
            units = "[0,1]",
            long_name = "Interpolated variance of coastline angles",
            description = "For each point, the variance of the coastline angles in the domain. This is a measure of how many coastlines are influencing a given point. A value of 0 indicates that coastlines are generally in agreement, and a value of 1 indicates that the point is influenced by coastlines in all directions. The variance is undefined at the coast, and here the variance is interpolated across the coastline."
            )
        
        angle_ds = angle_ds.assign_attrs(
            description = "Dataset of coastline angles and variance",
            acknowledgmements = "This method was developed with help from Ewan Short and Jarrah Harrison-Lofthouse.",
            R_km = str(R)
            )

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

    if smooth:
        angle_ds["angle_interp"] = smooth_angles(angle_ds["angle_interp"],sigma)

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

def interpolate_variance(angle_ds):

    """
    From a dataset of coastline variance, interpolate across the coastline.

    This is used because the result of get_coastline_angle_kernel() is not defined along the coastline.
    """

    xx,yy = np.meshgrid(angle_ds.lon,angle_ds.lat)

    points = angle_ds.variance.values.ravel()
    valid = ~np.isnan(points)
    points_valid = points[valid]
    xx_rav, yy_rav = xx.ravel(), yy.ravel()
    xxv = xx_rav[valid]
    yyv = yy_rav[valid]
    interpolated_variance = scipy.interpolate.griddata(np.stack([xxv, yyv]).T, points_valid, (xx, yy), method="linear").reshape(xx.shape)     
    interpolated_variance_da = xr.DataArray(interpolated_variance,dims=angle_ds.dims,coords=angle_ds.coords)

    angle_ds["variance_interp"] = interpolated_variance_da

    return angle_ds

def aus2200_hybrid_height_calc(height_coord="rho"):

    """
    This function calculates model level heights for the AUS2200 model, using either the eta_rho and eta_theta coordinates.
    Model level heights are saved to disk as a zarr file, which can be used for interpolation of model data to the aus2200 height levels.
    
    #Input
    * height_coord: string - either "rho" or "theta". This determines which coordinate to use for the hybrid height calculation. The default is "rho", for wind data
    """

    #From ~access/umdir/vn13.8/ctldata/vert/vertlevs_L70_50t_20s_80km
    #This is a list of AUS2200 model levels and associated metadata, that will be used to reconstruct the hybrid height coordinate
    first_constant_r_rho_level= 62
    if height_coord == "rho":
        eta=np.array([0.6249999E-04,   0.3333333E-03,   0.8333333E-03,   0.1500000E-02,   0.2333333E-02,
            0.3333333E-02,   0.4500000E-02,   0.5833333E-02,   0.7333333E-02,   0.9000000E-02,
            0.1083333E-01,   0.1283333E-01,   0.1500000E-01,   0.1733333E-01,   0.1983333E-01,
            0.2250000E-01,   0.2533333E-01,   0.2833333E-01,   0.3150000E-01,   0.3483333E-01,
            0.3833333E-01,   0.4200000E-01,   0.4583333E-01,   0.4983333E-01,   0.5400000E-01,
            0.5833334E-01,   0.6283334E-01,   0.6750000E-01,   0.7233334E-01,   0.7733333E-01,
            0.8250000E-01,   0.8783333E-01,   0.9333333E-01,   0.9900000E-01,   0.1048333E+00,
            0.1108333E+00,   0.1170000E+00,   0.1233333E+00,   0.1298333E+00,   0.1365000E+00,
            0.1433333E+00,   0.1503334E+00,   0.1575020E+00,   0.1648455E+00,   0.1723789E+00,
            0.1801272E+00,   0.1881285E+00,   0.1964356E+00,   0.2051189E+00,   0.2142687E+00,
            0.2239982E+00,   0.2344463E+00,   0.2457803E+00,   0.2581997E+00,   0.2719388E+00,
            0.2872708E+00,   0.3045112E+00,   0.3240212E+00,   0.3462124E+00,   0.3715503E+00,
            0.4005586E+00,   0.4338236E+00,   0.4719992E+00,   0.5158110E+00,   0.5660614E+00,
            0.6236348E+00,   0.6895022E+00,   0.7647274E+00,   0.8504717E+00,   0.9480625E+00])
    elif height_coord == "theta":
        eta=np.array([0.0000000E+00,   0.1250000E-03,   0.5416666E-03,   0.1125000E-02,   0.1875000E-02,
            0.2791667E-02,   0.3875000E-02,   0.5125000E-02,   0.6541667E-02,   0.8125000E-02,
            0.9875000E-02,   0.1179167E-01,   0.1387500E-01,   0.1612500E-01,   0.1854167E-01,
            0.2112500E-01,   0.2387500E-01,   0.2679167E-01,   0.2987500E-01,   0.3312500E-01,
            0.3654167E-01,   0.4012500E-01,   0.4387500E-01,   0.4779167E-01,   0.5187500E-01,
            0.5612501E-01,   0.6054167E-01,   0.6512500E-01,   0.6987500E-01,   0.7479167E-01,
            0.7987500E-01,   0.8512500E-01,   0.9054167E-01,   0.9612500E-01,   0.1018750E+00,
            0.1077917E+00,   0.1138750E+00,   0.1201250E+00,   0.1265417E+00,   0.1331250E+00,
            0.1398750E+00,   0.1467917E+00,   0.1538752E+00,   0.1611287E+00,   0.1685623E+00,
            0.1761954E+00,   0.1840590E+00,   0.1921980E+00,   0.2006732E+00,   0.2095645E+00,
            0.2189729E+00,   0.2290236E+00,   0.2398690E+00,   0.2516917E+00,   0.2647077E+00,
            0.2791699E+00,   0.2953717E+00,   0.3136506E+00,   0.3343919E+00,   0.3580330E+00,
            0.3850676E+00,   0.4160496E+00,   0.4515977E+00,   0.4924007E+00,   0.5392213E+00,
            0.5929016E+00,   0.6543679E+00,   0.7246365E+00,   0.8048183E+00,   0.8961251E+00,
            0.1000000E+01])
    else:
        raise ValueError("height_coord must be either 'rho' or 'theta'")        
    
    #Load the atmosphere_hybrid_height_coordinate variable as levs
    ua = xr.open_dataset(
        "/g/data/bs94/AUS2200/mjo-neutral2013/v1-0/1hr/ua/ua_AUS2200_mjo-neutral_1hrPt_201303011900-201303020000.nc",
        chunks={}
        )
    levs = ua.lev.values

    #Load the orography file
    orog = xr.open_dataset("/g/data/bs94/AUS2200/mjo-neutral2013/v1-0/fx/orog/orog_AUS2200_mjo-neutral_fx.nc")

    #Calculate C 
    C = (1 - eta / eta[first_constant_r_rho_level-1])**2
    C[first_constant_r_rho_level:] = 0

    #Z will be of shape lat,lon,lev, so repeat C over lat/lon dimensions
    C_repeated = np.repeat(
            np.repeat(
            C[np.newaxis,np.newaxis,0:len(levs)],
            orog.orog.shape[0],
            axis=0),
        orog.orog.shape[1],
        axis=1
    )

    #And repeat orography over the level dimension
    orog_repeated = np.repeat(
        orog.orog.values[:,:,np.newaxis],
        len(levs),
        axis=2)
    
    #Calculate the height above sea level
    Z = levs + C_repeated * orog_repeated

    #Convert to height above ground level and assign to a chunked dataarray
    Z_agl_da = xr.DataArray(Z - orog_repeated,
                dims=["lat","lon","lev"],
                coords={"lat":orog.lat,"lon":orog.lon,"lev":ua.lev}).chunk({
        "lev":-1,
        "lon":ua.ua.chunksizes["lon"][0],
        "lat":ua.ua.chunksizes["lat"][0]})

    #Assign attributes and save to disk
    xr.Dataset({"Z_agl":Z_agl_da.assign_attrs(
        {"description":"Height above surface for the AUS2200 grid",
        "units":"m"}
        )}).to_zarr("/g/data/ng72/ab4502/sea_breeze_detection/aus2200_z_agl.zarr",mode="w")    