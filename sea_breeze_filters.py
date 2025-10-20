import scipy.ndimage
import skimage
import scipy
import numpy as np
import pandas as pd
import xarray as xr
import datetime as dt
import metpy.calc as mpcalc
import dask.array as da
from sea_breeze import utils
import os
import logging

class Mask_Options:
    
    """
    Handles options for filtering sea breeze objects.

    For more details on each filter and threshold, see the filter_2d function documentation.

    Attributes
    ----------
    filters : dict
        Dictionary of boolean flags indicating which filters are active.
        Keys include:
            - "orientation_filter" (default: False)
            - "aspect_filter" (default: True)
            - "area_filter" (default: True)
            - "time_filter" (default: False)
            - "dist_to_coast_filter" (default: False)
            - "land_sea_temperature_filter" (default: False)
            - "temperature_change_filter" (default: False)
            - "humidity_change_filter" (default: False)
            - "wind_change_filter" (default: False)
            - "onshore_wind_filter" (default: False)

    thresholds : dict
        Dictionary of threshold values for various filters.
        Keys include:
            - "orientation_tol" : float
                Tolerance for orientation (degrees)
            - "area_thresh_km" : float
                Minimum area threshold (km^2)
            - "area_thresh_pixels" : float
                Minimum area threshold (pixels)
            - "aspect_thresh" : float
                Minimum aspect ratio
            - "hour_min_lst" : int
                Minimum local solar time (hour)
            - "hour_max_lst" : int
                Maximum local solar time (hour)
            - "land_sea_temperature_diff_thresh" : float
                Minimum land-sea temperature difference
            - "temperature_change_thresh" : float
                Maximum temperature change
            - "humidity_change_thresh" : float
                Minimum humidity change
            - "wind_change_thresh" : float
                Minimum wind speed change in onshore direction
            - "max_distance_to_coast_thresh" : float
                Maximum distance to coast (km)
            - "min_distance_to_coast_thresh" : float
                Minimum distance to coast (km)
            - "onshore_wind_thresh" : float
                Minimum onshore wind speed

    options : dict
        Dictionary of additional options.
        Keys include:
            - "land_sea_temperature_radius" : int
                Radius for rolling maximum filter for land-sea temperature difference (km).
            - "area_filter_units" : str
                Units for area filtering ('pixels' or 'kms').
    Methods
    -------
    set_options(kwargs)
        Update filter, threshold, or option values using keyword arguments.
    __str__()
        Return a string representation of the current filter, threshold, and option settings.
    """

    def __init__(self):
        self.filters = {
            "orientation_filter":False,
            "aspect_filter":True,
            "area_filter":True,
            "time_filter":False,
            "dist_to_coast_filter":False,
            "land_sea_temperature_filter":False,
            "temperature_change_filter":False,
            "humidity_change_filter":False,
            "wind_change_filter":False,
            "onshore_wind_filter":False
        }

        self.thresholds = {
            "orientation_tol":30,                     #Within 30 degrees of coastline
            "area_thresh_km":50,                      #Area of 50 km^2 or greater
            "area_thresh_pixels":20,                  #Area of 10 pixels or greater
            "aspect_thresh":3,                        #Aspect ratio of 3 or greater
            "hour_min_lst":9,                         #LST hour must be greater than or equal to 12
            "hour_max_lst":22,                        #LST hour must be less than or equal to 23
            "land_sea_temperature_diff_thresh":0,     #Land sea temperature difference must be greater than zero
            "temperature_change_thresh":0,            #Temperature change must be less than zero
            "humidity_change_thresh":0,               #Humidity change must be greater than zero
            "wind_change_thresh":0,                   #Wind speed change in onshore direction must be greater than zero
            "max_distance_to_coast_thresh":300,       #Within 300 km of the coast
            "min_distance_to_coast_thresh":0,         #At least 0 km from the coast
            "onshore_wind_thresh":0                   #Onshore wind speed must be greater than zero
            }       
        
        self.options = {
            "land_sea_temperature_radius":50,         #Radius of the rolling maximum filter for land sea temperature difference (km)
            "area_filter_units":"pixels"              #Is area filtered by pixels or kms? Either 'pixels' or 'kms'
        }

    def set_options(self,kwargs):
        for k, val in kwargs.items():
            if k in self.filters.keys():
                self.filters[k] = val
            elif k in self.thresholds.keys():
                self.thresholds[k] = val
            elif k in self.options.keys():
                self.options[k] = val
            else:
                raise ValueError(f"{k} not a valid option")
        return self

    def __str__(self):
        return f"Filters: {self.filters} \n Thresholds: {self.thresholds}"

def percentile(field,p=99.5,skipna=False):

    """
    Calculate the pth percentile of an xarray DataArray field over all dimensions.

    Parameters
    ----------
    field : xarray.DataArray
        Input data array.
    p : float, optional
        Percentile to compute (default is 99.5).
    skipna : bool, optional
        If True, skip NaN values using climtas (default is False).

    Returns
    -------
    percentile : dask.array.Array
        The computed percentile value.

    Notes
    -----
    Uses dask for computation by default. If skipna is True, uses climtas for NaN-safe calculation.
    """
    
    if skipna:
        import climtas
        #Re-shape field for percentile calculation.
        field_stacked = field.stack(z=list(field.dims))
        return da.array(climtas.blocked.approx_percentile(field_stacked, dim="z", q=p, skipna=True))
    else:
        field_stacked = da.array(field).flatten()
        return da.percentile(field_stacked,p,internal_method="tdigest")

def binary_mask(field,thresh):

    """
    Create a binary sea breeze mask from a diagnostic field using a threshold.

    Parameters
    ----------
    field : xarray.DataArray
        The input field diagnosing sea breeze potential.
    thresh : float
        Threshold value for mask generation.

    Returns
    -------
    xarray.Dataset
        Dataset containing the binary mask, with mask values of 1 where field > thresh and 0 elsewhere.
        The mask has an attribute 'threshold' set to the input threshold value.
    """
    mask = (field>thresh)*1
    mask = mask.assign_attrs({"threshold":thresh})
    return xr.Dataset({"mask":mask})

def initialise_props_df_output(props_df_out_path,props_df_template):
    """
    Initialises an output CSV file for storing object properties based on a template DataFrame.

    This function checks if the specified output file exists and removes it if necessary.
    It then creates a new CSV file with the same columns as the provided template DataFrame,
    but without any data, ready to store filtered object properties.

    Parameters
    ----------
    props_df_out_path : str
        Path to the output CSV file to be created.
    props_df_template : pandas.DataFrame
        Template DataFrame whose columns will be used for the output CSV file.
    """
    if os.path.exists(props_df_out_path):
        os.remove(props_df_out_path)
    pd.DataFrame(columns=props_df_template.columns).to_csv(props_df_out_path,index=False)

def process_time_slice(time_slice,**kwargs):
    """
    For using filter_2d with map_blocks
    """
    ds, df = filter_2d(time_slice.squeeze(), **kwargs)
    return ds

def circmean_wrapper(data):
    """
    Wrapper for circmean function to use with xarray apply_ufunc
    """
    return scipy.stats.circmean(data, low=-90, high=90)

def filter_2d(ds,angle_ds=None,lsm=None,props_df_output_path=None,output_land_sea_temperature_diff=False,**kwargs):
    
    """
    Take a binary sea breeze mask and identify objects, then filter it for sea breezes based on several conditions related to those objects.

    Works for a 2D "lat"/"lon" xarray dataset. Can be extended to work with a time dimension using map_blocks (see process_time_slice and filter_3d).

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with variable "mask", a 2D binary mask of sea breeze objects and attribute "time" (UTC). May also contain for additional filtering:
            - "ta" : surface temperature
            - "t_change" : temperature change
            - "q_change" : specific humidity change
            - "wind_change" : onshore wind speed change
            - "vprime" : onshore wind component
    angle_ds : xarray.Dataset, optional
        Dataset of coastline angles and distance to coast for orientation and distance to coast filtering. Must contain:
            - "angle_interp" : interpolated coastline angle
            - "min_coast_dist" : minimum distance to coast
    lsm : xarray.DataArray, optional
        Land-sea mask.
    props_df_output_path : str, optional
        Path to output a CSV file of the properties of each object. If None, defaults to `./props_df.csv`.
    output_land_sea_temperature_diff : bool, optional
        If True, output the land-sea temperature difference as a DataArray. Default is False.
    **kwargs
        Options for filtering the sea breeze objects. See below. See Mask_Options class for details and defaults.

    Keywords in kwargs
    ----------------
    orientation_filter : bool, optional
        Filter objects based on orientation relative to the coastline. Requires angle_ds.
    aspect_filter : bool, optional
        Filter objects based on aspect ratio.
    area_filter : bool, optional
        Filter objects based on area.
    dist_to_coast_filter : bool, optional
        Filter objects based on distance to the coast. Requires angle_ds.
    time_filter : bool, optional
        Filter objects based on local solar time.
    land_sea_temperature_filter : bool, optional
        Filter objects based on land-sea temperature difference. Requires "ta" and lsm.
    temperature_change_filter : bool, optional
        Filter objects based on mean local temperature decrease. Requires "t_change".
    humidity_change_filter : bool, optional
        Filter objects based on mean local humidity increase. Requires "q_change".
    wind_change_filter : bool, optional
        Filter objects based on mean local onshore wind speed increase. Requires "wind_change".
    onshore_wind_filter : bool, optional
        Filter objects based on onshore wind speed. Requires "vprime".
    orientation_tol : float, optional
        Tolerance in degrees for orientation filter.
    area_thresh_km : float, optional
        Minimum area in km^2 for area filter if area_filter_units='kms'.
    area_thresh_pixels : float, optional
        Minimum area in pixels for area filter if area_filter_units='pixels'.
    aspect_thresh : float, optional
        Minimum aspect ratio for aspect filter.
    hour_min_lst : int, optional
        Minimum local solar time in hours for time filter.
    hour_max_lst : int, optional
        Maximum local solar time in hours for time filter.
    land_sea_temperature_diff_thresh : float, optional
        Minimum land-sea temperature difference for land-sea temperature filter.
    max_distance_to_coast_thresh : float, optional
        Maximum distance to coast in km for distance to coast filter.
    min_distance_to_coast_thresh : float, optional
        Minimum distance to coast in km for distance to coast filter.
    temperature_change_thresh : float, optional
        Maximum temperature change for temperature change filter.
    humidity_change_thresh : float, optional
        Minimum humidity change for humidity change filter.
    wind_change_thresh : float, optional
        Minimum wind change for onshore wind change filter.
    onshore_wind_thresh : float, optional
        Minimum onshore wind speed for onshore wind filter.
    land_sea_temperature_radius : float, optional
        Radius of the rolling maximum/mean filter for land-sea temperature difference, in km.
    area_filter_units : {'pixels', 'kms'}, optional
        Units for area filtering.

    Returns
    -------
    ds : xarray.Dataset
        Binary mask of sea breeze objects after filtering.
    props_df : pandas.DataFrame
        DataFrame of object properties.    
    """

    if ds.mask.ndim > 2:
        raise ValueError("mask must be a 2D array (of lat and lon). If you have a time dimension, use map_blocks to process each time slice. See process_time_slice function or filter_ds_driver")

    #Set up logging
    logging.getLogger("flox").setLevel(logging.WARNING)

    #Set options
    mask_options = Mask_Options().set_options(kwargs)

    #Get time for data array
    #time = pd.to_datetime(mask.time.values).strftime("%Y-%m-%d %H:%M")
    time = ds.time.values

    #Check for missing values in each ds variable, and raise a warning if found
    for var in ds.data_vars:
        if ds[var].isnull().any():
            logging.warning(f"Warning: Missing values found in {var} for time {time}. This may affect filtering.")

    #From a binary (mask) array of candidate sea breeze objects, label from 1 to N
    labels = skimage.measure.label(ds["mask"])
    labels_da = xr.DataArray(labels, dims=ds["mask"].dims, coords=ds["mask"].coords)

    #Using skimage, return properties for each candidate object
    region_props = skimage.measure.regionprops(labels,spacing=(1,1))

    #Get longitudes of image for the purpose of converting to local solar time
    lons = ds.lon.values
    lats = ds.lat.values

    #Get area of pixels using metpy
    dx,dy,pixel_area = utils.metpy_grid_area(ds.lon,ds.lat)
    pixel_area = xr.DataArray(pixel_area,coords=ds["mask"].coords,dims=ds["mask"].dims)

    #For each object create lists of relevant properties
    labs = np.array([region_props[i].label for i in np.arange(len(region_props))])                                    #Object label number
    eccen = [region_props[i].eccentricity for i in np.arange(len(region_props))]                                      #Eccentricity
    area = [region_props[i].area for i in np.arange(len(region_props))]                                               #Area
    orient = np.rad2deg(np.array([region_props[i].orientation for i in np.arange(len(region_props))]))                #Orientation angle
    major = np.array([region_props[i].axis_major_length for i in np.arange(len(region_props))])                       #Major axis length in pixels
    minor = np.array([region_props[i].axis_minor_length for i in np.arange(len(region_props))])                       #Minor axis length in pixels
    centroid_lons = [lons[np.round(region_props[i].centroid[1]).astype(int)] for i in np.arange(len(region_props))]  
    centroid_lats = [lats[np.round(region_props[i].centroid[0]).astype(int)] for i in np.arange(len(region_props))]  
    lst = [pd.to_datetime(time) + dt.timedelta(hours=l / 180 * 12) for l in centroid_lons]                            #Local solar time based on centroid longitude

    #Create pandas dataframe with object properties, index by label number
    props_df = pd.DataFrame({
        "eccen":eccen,
        "area":area,
        "orient":orient,
        "major":major,
        "minor":minor,
        "lon_centroid":centroid_lons,
        "lat_centroid":centroid_lats,
        "lst":pd.to_datetime(lst)}, index=labs)        
    props_df["aspect"] = props_df.major/props_df.minor
    
    #Create condition to keep objects
    cond = pd.DataFrame(np.ones(props_df.shape[0]).astype(bool),index=props_df.index)[0]

    #Filter objects based on orientation relative to coastlines
    if mask_options.filters["orientation_filter"]:
        if (angle_ds is not None):
            #Calculate the mean angle of nearby coastlines over each labelled region using xarray apply ufunc and scipy circmean
            mean_angles = angle_ds.angle_interp.groupby(labels_da.rename("label")).map(
                    lambda x: xr.apply_ufunc(
                        circmean_wrapper,
                        x,
                        input_core_dims=[['stacked_lat_lon']],
                        vectorize=True
                    )
                ).to_series().drop(0)    
            cond = cond & ( (props_df.orient - mean_angles).abs() <= mask_options.thresholds["orientation_tol"])
        else:
            raise ValueError("angle_ds must be provided for orientation filter")

    #Filter objects based on aspect ratio
    if mask_options.filters["aspect_filter"]:
        cond = cond & ( props_df.aspect >= mask_options.thresholds["aspect_thresh"] )

    #Filter objects based on area
    if mask_options.filters["area_filter"]:
        mean_pixel_area = pixel_area.groupby(labels_da.rename("label")).mean().to_series().drop(0)
        props_df["area_km"] = props_df.area*mean_pixel_area
        if mask_options.options["area_filter_units"] == "pixels":
            cond = cond & ( (props_df.area) >= mask_options.thresholds["area_thresh_pixels"] )
        elif mask_options.options["area_filter_units"] == "kms":
            cond = cond & ( (props_df.area_km) >= mask_options.thresholds["area_thresh_km"] )
        else:
            raise ValueError("'area_filter_units' must be either 'kms' or 'pixels'")

    #Filter objects based on distance to coast
    if mask_options.filters["dist_to_coast_filter"]:
        if (angle_ds is not None):
            #Calculate the mean distance to coastline over each labelled region
            mean_dist = angle_ds.min_coast_dist.groupby(labels_da.rename("label")).mean().to_series().drop(0)
            props_df["mean_dist_to_coast_km"] = mean_dist
            cond = cond & ( mean_dist <= mask_options.thresholds["max_distance_to_coast_thresh"] )
            cond = cond & ( mean_dist >= mask_options.thresholds["min_distance_to_coast_thresh"] )
        else:
            raise ValueError("angle_ds must be provided for distance to coast filter")

    #Filter objects based on local solar time
    if mask_options.filters["time_filter"]:
        cond = cond & ( props_df.lst.dt.hour >= mask_options.thresholds["hour_min_lst"] ) &\
            ( props_df.lst.dt.hour <= mask_options.thresholds["hour_max_lst"] )
        
    #Filter objects based on land sea temperature difference
    if mask_options.filters["land_sea_temperature_filter"]:
        if ("ta" in list(ds.keys())) & (lsm is not None):
            #Calculate the local (smoothed) difference between nearest land and sea temperatures
            #Here use a radius of 50 km for the smoothing (radial max over land, mean over ocean)
            land_sea_temp_diff = land_sea_temperature_diff_rolling_max(
                ds["ta"],
                lsm,
                R_km=mask_options.options["land_sea_temperature_radius"],
                dy=dy,dx=dx)

            #Calculate the mean land-sea temperature difference over each labelled region
            mean_land_sea_temp_diff = land_sea_temp_diff.groupby(labels_da.rename("label")).mean().to_series().drop(0)
            props_df["mean_land_sea_temp_diff"] = mean_land_sea_temp_diff
            cond = cond & ( mean_land_sea_temp_diff > mask_options.thresholds["land_sea_temperature_diff_thresh"] ) 
        else:
            raise ValueError("ta must be in ds and lsm must be provided for land sea temperature filter")   
    
    #Filter objects based on local temperature decrease
    if mask_options.filters["temperature_change_filter"]:
        if ("t_change" in list(ds.keys())):
            #Calculate the mean temperature change over each labelled region
            mean_t_change = ds["t_change"].groupby(labels_da.rename("label")).mean().to_series().drop(0)
            props_df["mean_t_change"] = mean_t_change
            cond = cond & ( mean_t_change < mask_options.thresholds["temperature_change_thresh"] ) 
        else:
            raise ValueError("t_change must be provided in ds for temperature change filter")
        
    #Filter objects based on local humidity increase
    if mask_options.filters["humidity_change_filter"]:
        if ("q_change" in list(ds.keys())):
            #Calculate the mean humidity change over each labelled region
            mean_q_change = ds["q_change"].groupby(labels_da.rename("label")).mean().to_series().drop(0)
            props_df["mean_q_change"] = mean_q_change
            cond = cond & ( mean_q_change > mask_options.thresholds["humidity_change_thresh"] ) 
        else:
            raise ValueError("q_change must be provided in ds for humidity change filter")
        
    #Filter objects based on local onshore wind speed increase
    if mask_options.filters["wind_change_filter"]:
        if ("wind_change" in list(ds.keys())):
            #Calculate the mean wind change over each labelled region
            mean_wind_change = ds["wind_change"].groupby(labels_da.rename("label")).mean().to_series().drop(0)
            props_df["mean_wind_change"] = mean_wind_change
            cond = cond & ( mean_wind_change > mask_options.thresholds["wind_change_thresh"] ) 
        else:
            raise ValueError("wind_change must be provided in ds for wind change filter")

    #Filter objects based on onshore wind speed
    if mask_options.filters["onshore_wind_filter"]:
        if ("vprime" in list(ds.keys())):
            #Calculate the mean onshore wind speed over each labelled region
            mean_vprime = ds["vprime"].groupby(labels_da.rename("label")).mean().to_series().drop(0)
            props_df["mean_vprime"] = mean_vprime
            cond = cond & ( mean_vprime > mask_options.thresholds["onshore_wind_thresh"] ) 
        else:
            raise ValueError("vprime must be provided in ds for onshore wind speed filter")

    #Subset label dataframe based on these conditions
    props_df = props_df.loc[cond]
    props_df["time_utc"] = time

    #Remove labels that don't meet conditions from mask
    mask = xr.where(labels_da.isin(props_df.loc[cond].index),1,0)

    #Assign attributes to mask
    for key, val in mask_options.filters.items():
        mask = mask.assign_attrs({key:str(val)})
    for key, val in mask_options.thresholds.items():
        mask = mask.assign_attrs({key:str(val)})
    for key, val in mask_options.options.items():
        mask = mask.assign_attrs({key:str(val)})
    
    #Convert to a bool for less memory usage
    mask = mask.astype(bool)

    #Combine mask with label array and props_df
    ds = xr.Dataset({"mask":mask.assign_coords({"time":time}).expand_dims("time")})

    #Option to add land sea temperature difference to the dataset
    if (output_land_sea_temperature_diff) & (mask_options.filters["land_sea_temperature_filter"]):
        ds = xr.merge((ds,land_sea_temp_diff.expand_dims("time").rename("land_sea_temp_diff")))

    #Save props_df to csv and output with label index as a column
    if props_df_output_path is None:
        props_df_output_path = "/scratch/gb02/ab4502/tmp/props_df.csv"
    props_df = props_df.reset_index().rename({"index":"label"})
    props_df.to_csv(props_df_output_path,mode="a",header=False,index=False)

    return ds, props_df

def filter_3d(field,threshold="percentile",threshold_value=None,p=95,hourly_change_ds=None,ta=None,vprime=None,lsm=None,angle_ds=None,save_mask=False,filter_out_path=None,props_df_out_path=None,skipna=False,output_chunks=None,**kwargs):

    """
    Identify sea breeze objects.

    Takes a sea breeze diagnostic field, creates a binary mask, then filters that mask for sea breeze objects based on several conditions (filters) using the filter_2d() function. Works for a 3D lat/lon/time field as an xarray dataset. A binary mask is created by taking exceedances of a threshold, which can either be "fixed" (provided by the user) or a percentile value calculated from the dataset. The filtering is applied using map_blocks to each time slice, and therefore the field is re-chunked in time.

    Parameters
    ----------
    field : xarray.DataArray
        Sea breeze diagnostic field with dimensions ('lat', 'lon', 'time').
    threshold : {'percentile', 'fixed'}, optional
        Threshold method for creating binary sea breeze mask. 'percentile' calculates the Pth percentile from 'field', 'fixed' uses 'threshold_value' provided by the user.
    threshold_value : float, optional
        Threshold for creating a binary sea breeze mask if threshold='fixed'.
    p : float, optional
        Percentile used to create a binary mask from field if threshold='percentile', between 0 and 100.
    hourly_change_ds : xarray.Dataset, optional
        Dataset containing "t_change" (temperature change), "q_change" (specific humidity change), and/or "wind_change" (onshore wind speed change) as variables. Needed for temperature_change_filter, humidity_change_filter, wind_change_filter (filter options in kwargs)
    ta : xarray.DataArray, optional
        Surface temperature, with the same coords and dims as field. Needed for land_sea_temperature_filter (filter options in kwargs).
    vprime : xarray.DataArray, optional
        Onshore wind speed, with the same coords and dims as field. Needed for onshore_wind_filter (filter options in kwargs).
    lsm : xarray.DataArray, optional
        Land-sea mask. Needed for land_sea_temperature_filter (filter options in kwargs).
    angle_ds : xarray.Dataset, optional
        Dataset of coastline angles created by sea_breeze_utils.get_coastline_angles. Needed for orientation_filter and dist_to_coast_filter (filter options in kwargs). Assumes variables "angle_interp" and "min_coast_dist".
    save_mask : bool, optional
        Whether to save the output mask dataset to disk.
    filter_out_path : str, optional
        If save_mask=True, path to save the filtered mask output.
    props_df_out_path : str, optional
        Path to output a CSV file of the properties of each object. Default is None, which sets it to "/scratch/gb02/ab4502/tmp/props_df.csv".
    skipna : bool, optional
        If True, calculation of the field percentile will ignore NaNs.
    output_chunks : dict, optional
        Chunking to use for zarr output. If None, uses default chunking {"time":1,"lat":-1,"lon":-1}.
    **kwargs
        Options for filtering the sea breeze objects passed to filter_2d. See filter_2d and Mask_Options class for details.

    Returns
    -------
    filtered_mask : xarray.Dataset
        Binary mask of sea breeze objects after filtering.

    Notes
    -----
    - The land_sea_temperature_filter can slow down the filtering significantly due to many nearest-neighbour lookups.
    - The land_sea_temperature_radius is converted to pixels using the mean grid spacing in x and y, and will therefore vary slightly with latitude.
    """

    if save_mask:
        if filter_out_path is None:
            raise ValueError("filter_out_path must be provided if save_mask is True")

    #Rechunk the field in time, as we are using map_blocks
    field = field.chunk({"time":1,"lat":-1,"lon":-1})

    #Create a binary mask from the field. Either by computing a percentile value ot using a fixed threshold
    if threshold=="percentile":
        print("INFO: Computing "+str(p)+"th percentile from field...")
        thresh = percentile(field,p=p,skipna=skipna)
        thresh = thresh.compute()
        print("Using threshold: ",str(thresh))
    elif threshold=="fixed":
        if threshold_value is not None:
            thresh = threshold_value
        else:
            raise ValueError("threshold is set to fixed but no threshold_value is provided")
    else:
        raise ValueError("threshold must be 'percentile' or 'fixed'")
    ds = binary_mask(field, thresh)

    #If angle_ds is provided, add to kwargs to pass to filter_2d
    if angle_ds is not None:
        kwargs["angle_ds"] = angle_ds.compute()

    #If lsm is provided, add to kwargs to pass to filter_2d
    if lsm is not None:
        kwargs["lsm"] = lsm.compute()
    
    #If ta is provided, re-chunk in time and combine with the mask dataset
    if ta is None:
        pass
    else:
        ta=ta.chunk({"time":1,"lat":-1,"lon":-1})
        ds = xr.merge((ds,ta.rename("ta")),join="left")  

    #If vprime is provided, re-chunk in time and combine with the mask dataset
    if vprime is None:
        pass
    else:
        vprime = vprime.chunk({"time":1,"lat":-1,"lon":-1})
        ds = xr.merge((ds,vprime.rename("vprime")),join="left") 

    #If hourly change data is provided, re-chunk in time and combine with the mask dataset
    if hourly_change_ds is None:
        pass
    else:
        hourly_change_ds = hourly_change_ds.chunk({"time":1,"lat":-1,"lon":-1})
        ds = xr.merge((ds,hourly_change_ds),join="left")         

    #We will apply the filtering using map_blocks.
    #First, need create a "template" from the first time step
    if props_df_out_path is None:
        props_df_out_path = "/scratch/gb02/ab4502/tmp/props_df.csv"
    kwargs["props_df_output_path"] = props_df_out_path
    template,props_df_template = filter_2d(ds.isel(time=0), **kwargs)
    template = template.reindex({"time":ds.time},fill_value=False).chunk({"time":1})

    #Setup the output dafaframe for saving sea breeze object properties as csv files
    initialise_props_df_output(props_df_out_path,props_df_template)

    #Apply the filtering
    filtered_mask = ds.map_blocks(
        process_time_slice,        
        template=template,
        kwargs=kwargs)

    #Set some extra attributes
    filtered_mask["mask"] = filtered_mask["mask"].assign_attrs({"threshold":ds.mask.threshold})
    filtered_mask["mask"] = filtered_mask["mask"].assign_attrs({"threshold_method":threshold})
    if threshold=="percentile":
        filtered_mask["mask"] = filtered_mask["mask"].assign_attrs({"percentile":str(p)})
    
    #Save the filtered mask if required
    if save_mask:
        drop_vars = ["crs","height","level_height","model_level_number","sigma"]
        for v in drop_vars:
            if v in list(filtered_mask.coords.keys()):
                filtered_mask = filtered_mask.drop_vars(v)        

        if output_chunks is None:
            mask_save = filtered_mask.to_zarr(filter_out_path,compute=False,mode="w")
        else:
            mask_save = filtered_mask.chunk(output_chunks).to_zarr(filter_out_path,compute=False,mode="w").persist()
    
    return filtered_mask

def land_sea_temperature_diff_rolling_max(ts,lsm,R_km,dy,dx):

    """
    Calculate the land-sea temperature difference for each point in the domain.

    For each point, the land temperature is filtered with a rolling maximum over a radius of R km, considering land points only.
    Then, for each point, the land-sea temperature difference is calculated as the difference between the closest land and closest sea point.

    Parameters
    ----------
    ts : xarray.DataArray
        Surface temperature.
    lsm : xarray.DataArray
        Land-sea mask.
    R_km : float
        Target radius of the rolling maximum filter in km. The actual radius is calculated by dividing this number by the mean grid spacing in x and y.
    dy : float or xarray.DataArray
        Y grid spacing in km.
    dx : float or xarray.DataArray
        X grid spacing in km.

    Returns
    -------
    xarray.DataArray
        Land-sea temperature difference for each point in the domain.
    """

    #First define land and sea nearest neighbour lookup for all points
    lat = lsm.lat.values
    lon = lsm.lon.values
    xx,yy = np.meshgrid(lon,lat)

    #Land NN
    land_x, land_y = np.where(lsm==1)
    land_lon = lon[land_y]
    land_lat = lat[land_x]
    land_X = np.array([land_lat, land_lon]).T
    land_kdt = scipy.spatial.KDTree(land_X)

    #Sea NN
    sea_x, sea_y = np.where(lsm==0)
    sea_lon = lon[sea_y]
    sea_lat = lat[sea_x]
    sea_X = np.array([sea_lat, sea_lon]).T
    sea_kdt = scipy.spatial.KDTree(sea_X)

    #Now look up the N closest land/sea points
    _,land_ind = land_kdt.query(np.array([yy.flatten(),xx.flatten()]).T, 1)
    target_lon_land = land_lon[land_ind.reshape((lat.shape[0],lon.shape[0]))]
    target_lat_land = land_lat[land_ind.reshape((lat.shape[0],lon.shape[0]))]
    target_lon_land = xr.DataArray(target_lon_land,dims=("lat","lon"),coords={"lat":lsm.lat,"lon":lsm.lon})
    target_lat_land = xr.DataArray(target_lat_land,dims=("lat","lon"),coords={"lat":lsm.lat,"lon":lsm.lon})

    #For sea NN
    _,sea_ind = sea_kdt.query(np.array([yy.flatten(),xx.flatten()]).T, 1)
    target_lon_sea = sea_lon[sea_ind.reshape((lat.shape[0],lon.shape[0]))]
    target_lat_sea = sea_lat[sea_ind.reshape((lat.shape[0],lon.shape[0]))]
    target_lon_sea = xr.DataArray(target_lon_sea,dims=("lat","lon"),coords={"lat":lsm.lat,"lon":lsm.lon})
    target_lat_sea = xr.DataArray(target_lat_sea,dims=("lat","lon"),coords={"lat":lsm.lat,"lon":lsm.lon})

    #Filter tas with a rolling mean and maximum over a radius
    def create_elliptical_footprint(radius_y, radius_x):
        """Create an elliptical footprint with given radii for y and x directions."""
        y, x = np.ogrid[-radius_y:radius_y+1, -radius_x:radius_x+1]
        footprint = (x**2 / radius_x**2) + (y**2 / radius_y**2) <= 1
        return footprint   
    radius_y = int(np.round(R_km / np.mean(dy.data)))
    radius_x = int(np.round(R_km / np.mean(dx.data)))
    footprint = create_elliptical_footprint(radius_y,radius_x)    
    #print(f"INFO: Radius for land_sea_temperature_difference smoothing in longitude direction ({radius_x} pixels) will vary with latitude, between {radius_x * np.min(dx.data)} km and {radius_x * np.max(dx.data)} km.\nINFO: Radius in latitude direction is {radius_y * np.mean(dy.data)} km ({radius_y} pixels)")
    

    land_ts = xr.DataArray(
        scipy.ndimage.maximum_filter(ts.where(lsm==1,-999),footprint=footprint),
        coords=ts.coords, dims=ts.dims)  

    #Function to index the ts DataArray using the target latitudes and longitudes
    def index_ts_dataarray(ts, target_lat, target_lon):
        """
        Index the ts DataArray using the target latitudes and longitudes.
        
        Parameters:
        * ts: xarray DataArray of the time series data
        * target_lat: 2D array of target latitudes
        * target_lon: 2D array of target longitudes
        """
        # Convert target lat/lon to indices
        lat_idx = xr.DataArray(np.searchsorted(ts.lat, target_lat))
        lon_idx = xr.DataArray(np.searchsorted(ts.lon, target_lon))

        # Use advanced indexing to get the values
        indexed_ts = ts.isel(lat=lat_idx, lon=lon_idx).values
        
        return indexed_ts
    
    land_selected = xr.DataArray(index_ts_dataarray(land_ts,target_lat_land,target_lon_land),dims=ts.dims,coords=ts.coords)
    sea_selected = xr.DataArray(index_ts_dataarray(ts,target_lat_sea,target_lon_sea),dims=ts.dims,coords=ts.coords)

    land_minus_sea = land_selected - sea_selected

    return land_minus_sea