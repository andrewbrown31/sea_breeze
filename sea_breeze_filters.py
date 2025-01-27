import scipy.ndimage
import skimage
import scipy
import tqdm
import numpy as np
import pandas as pd
import xarray as xr
import datetime as dt
import metpy.calc as mpcalc
import dask.array as da
from sea_breeze import utils
import os

class Mask_Options:
    
    '''
    Options for filtering sea breeze candidate objects
    '''

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
            "wind_change_filter":False
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
            "distance_to_coast_thresh":300}           #Within 300 km of the coast
        
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

def percentile(field,p=95,skipna=False):

    """
    From an xarray dataarray, calculate the pth percentile of the field
    
    The default here is to use dask. But if we want to skip nans, then the easier option is to use the climtas module.
    """
    #Re-shape field for percentile calculation.
    field_stacked = field.stack(z=list(field.dims))
    
    if skipna:
        import climtas
        return da.array(climtas.blocked.approx_percentile(field_stacked, dim="z", q=p, skipna=True))
    else:
        return da.percentile(da.array(field_stacked),p)

def binary_mask(field,thresh):

    """
    Take a field that diagnoses sea breeze potential and create a binary sea breeze mask based on a threshold

    Absolute versus percentile/perturbation thresholds? Standard deviation threshold
    """
    mask = (field>thresh)*1
    mask = mask.assign_attrs({"threshold":thresh})
    return xr.Dataset({"mask":mask})

def fuzzy_function(x, x1=0, y1=0, y2=1, D=2):

    """
    Fuzzy logic function from Coceal et al. 2018 (https://doi.org/10.1002/asl.846)

    ## Input
    * x: xarray dataarray of input values
    * x1: x value at which the function starts to increase
    * y1: y value for x <= x1
    * y2: y value for x >= x2, where x2 is the maximum value of x divided by D
    * D: scaling factor for x2
    """

    x2 = (x.max() / D)
    
    f_x = y1 + ( ( y2 - y1 ) / ( x2 - x1 ) ) * ( x - x1 )
    f_x = xr.where(x <= x1, y1, f_x)
    f_x = xr.where(x >= x2, y2, f_x)

    return f_x

def fuzzy_function_combine(wind_change,q_change,t_change,combine_method="product"):

    """
    From Coceal et al. 2018, a fuzzy logic method for identifying sea breezes. 

    ## Input
    * wind_change: xarray dataarray of wind speed change in the onshore direction
    * q_change: xarray dataarray of specific humidity change
    * t_change: xarray dataarray of temperature change
    * combine_method: method for combining the fuzzy functions. Can be "product" or "mean". Note that Coceal et al. 2018 use the mean method.

    ## Output
    * mask: binary mask of candidate sea breeze objects

    Note that t_change is assumed to be negative. For example, a sea breeze front results in a negative temperature change.

    Coceal, O., Bohnenstengel, S. I., & Kotthaus, S. (2018). Detection of sea-breeze events around London using a fuzzy-logic algorithm. Atmospheric Science Letters, 19(9). https://doi.org/10.1002/asl.846
    """

    #Calculate the fuzzy functions for each variable
    wind_fuzzy = fuzzy_function(wind_change)
    q_fuzzy = fuzzy_function(q_change)
    t_fuzzy = fuzzy_function(-t_change)

    #Combine the fuzzy functions
    if combine_method=="product":
        mask = (wind_fuzzy * q_fuzzy * t_fuzzy)
    elif combine_method=="mean":
        mask = ((wind_fuzzy + q_fuzzy + t_fuzzy) / 3)
    else:
        raise ValueError("combine_method must be 'product' or 'mean'")

    mask = mask.assign_attrs({"combine_method":combine_method})
    mask = mask.assign_attrs(
        units = "[0,1]",
        long_name = "Fuzzy sea breeze detection algorithm",
        description = "Fuzzy sea breeze detection algorithm using the rate of change of moisture, temperature and onshore wind speed, following Coceal et al. (2018)")      

    return mask

def initialise_props_df_output(props_df_out_path,props_df_template):
    """
    When applying the filtering using map_blocks, we need to create an output dataframe to store the object properties.
    """
    if os.path.exists(props_df_out_path):
        os.remove(props_df_out_path)
    pd.DataFrame(columns=props_df_template.columns).to_csv(props_df_out_path,index=False)

def process_time_slice(time_slice,**kwargs):
    """
    For using filter_ds with map_blocks
    """
    ds, df = filter_ds(time_slice.squeeze(), **kwargs)
    return ds

def circmean_wrapper(data):
    """
    Wrapper for circmean function to use with xarray apply_ufunc
    """
    return scipy.stats.circmean(data, low=-90, high=90)

def filter_ds(ds,angle_ds=None,lsm=None,props_df_output_path=None,output_land_sea_temperature_diff=False,**kwargs):
    
    """
    ## Description
    Take a binary sea breeze mask and identify objects, then filter it for sea breezes based on several conditions related to those objects. Works for a 2d lat/lon xarray dataset, but can be extended to work with a time dimension using map_blocks (see process_time_slice function and filter_ds_driver).

    ## Input
    ### Required
    * ds: xarray dataset with a variable "mask", containing a 2d binary mask of sea breeze objects, as an xarray dataarray. May also contain the following variables with the same shape and coordinates as mask as xarray dataarrays:
        * "ta" (surface temperature)
        * "t_change" (temperature change, see sea_breeze_funcs.hourly_change)
        * "q_change" (specific humidity change, see sea_breeze_funcs.hourly_change)
        * "wind_change" (onshore wind speed change, see sea_breeze_funcs.hourly_change)

    ### Optional
    * angle_ds: xarray dataset of coastline angles created by get_coastline_angles.py
    * lsm: xarray dataarray of the land-sea mask
    * props_df_output_path: path to output a csv file of the properties of each object
    * output_land_sea_temperature_diff: output the land sea temperature difference as a dataarray (bool, default False)
    * **kwargs: options for filtering the sea breeze objects (see below)

    ## Options (kwargs)
    ### Filters
    * orientation_filter: Filter candidate objects based on orientation parallel to the coastline True/False (default False)
         - Requires angle_ds to be provided
    * aspect_filter: Filter candidate objects based on aspect ratio True/False (default True)
    * area_filter: Filter candidate objects based on area True/False (default True)
    * dist_to_coast_filter: Filter candidate objects based on distance to the coast True/False (default False)
        - Requires angle_ds to be provided
    * time_filter: Filter candidate objects based on local solar time True/False (default True)
    * land_sea_temperature_filter: Filter candidate objects based on land sea temperature difference True/False (default False)
        - Requires ta as a variable in ds, and lsm to be provided
    * temperature_change_filter: Filter candidate objects based on mean local temperature decrease True/False (default False)
        - Requires t_change to be provided as a variable in ds
    * humidity_change_filter: Filter candidate objects based on mean local humidity increase True/False (default False)
        - Requires q_change to be provided as a variable in ds
    * wind_change_filter: Filter candidate objects based on mean local onshore wind speed increase True/False (default False)
        - Requires wind_change to be provided as a variable in ds

    ### Thresholds
    * orientation_tol: Tolerance in degrees for orientation filter (default 30 degrees)
    * area_thresh: Minimum area in km^2 for area filter if area_filter_units='kms' (defailt 50 km^2)
    * area_thresh_pixels: Minimum area in pixels for area filter if area_filter_units='pixels' (defailt 20)
    * aspect_thresh: Minimum aspect ratio for aspect filter (default 3)
    * hour_min_lst: Minimum local solar time in hours for time filter (default 9)
    * hour_max_lst: Maximum local solar time in hours for time filter (default 22)
    * land_sea_temperature_diff_thresh: Minimum land sea temperature difference for land sea temperature filter (default 0)
    * distance_to_coast_thresh: Maximum distance to coast in km for distance to coast filter (default 300 km)
    * temperature_change_thresh: Maximum temperature change for temperature change filter, same units as t_change (default 0)
    * humidity_change_thresh: Minimum humidity change for humidity change filter, same units as q_change (default 0)
    * wind_change_thresh: Minimum wind change for onshore wind change filter, same units as wind_change (default 0)

    ### Other options
    * land_sea_temperature_radius: Radius of the rolling maximum/mean filter for land sea temperature difference, in km (default 50 km). NOTE that this radius is converted to pixels by using the mean grid spacing in x and y, and will therefore vary with latitude.
    * area_filter_units: Is area filtered by pixels or kms? Either 'pixels' or 'kms'

    ## Output
    * mask: binary mask of sea breeze objects after filtering

    TODO Propagation speed or persistance? Note this is not straightforward. Will need to be in a separate function with input having a time dimension.

    """

    if ds.mask.ndim > 2:
        raise ValueError("mask must be a 2D array (of lat and lon). If you have a time dimension, use map_blocks to process each time slice. See process_time_slice function or filter_ds_driver")

    #Set options
    mask_options = Mask_Options().set_options(kwargs)

    #Get time for data array
    #time = pd.to_datetime(mask.time.values).strftime("%Y-%m-%d %H:%M")
    time = ds.time.values

    #From a binary (mask) array of candidate sea breeze objects, label from 1 to N
    labels = skimage.measure.label(ds["mask"])
    labels_da = xr.DataArray(labels, dims=ds["mask"].dims, coords=ds["mask"].coords)

    #Using skimage, return properties for each candidate object
    region_props = skimage.measure.regionprops(labels,spacing=(1,1))

    #Get longitudes of image for the purpose of converting to local solar time
    lons = ds.lon.values

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
    lst = [pd.to_datetime(time) + dt.timedelta(hours=l / 180 * 12) for l in centroid_lons]                            #Local solar time based on centroid longitude

    #Create pandas dataframe with object properties, index by label number
    props_df = pd.DataFrame({
        "eccen":eccen,
        "area":area,
        "orient":orient,
        "major":major,
        "minor":minor,
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
                ).to_series()    
            cond = cond & ( (props_df.orient - mean_angles).abs() <= mask_options.thresholds["orientation_tol"])
        else:
            raise ValueError("angle_ds must be provided for orientation filter")

    #Filter objects based on aspect ratio
    if mask_options.filters["aspect_filter"]:
        cond = cond & ( props_df.aspect >= mask_options.thresholds["aspect_thresh"] )

    #Filter objects based on area
    if mask_options.filters["area_filter"]:
        mean_pixel_area = pixel_area.groupby(labels_da.rename("label")).mean().to_series()
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
            mean_dist = angle_ds.min_coast_dist.groupby(labels_da.rename("label")).mean().to_series()
            props_df["mean_dist_to_coast_km"] = mean_dist
            cond = cond & ( mean_dist <= mask_options.thresholds["distance_to_coast_thresh"] )
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
            mean_land_sea_temp_diff = land_sea_temp_diff.groupby(labels_da.rename("label")).mean().to_series()
            props_df["mean_land_sea_temp_diff"] = mean_land_sea_temp_diff
            cond = cond & ( mean_land_sea_temp_diff > mask_options.thresholds["land_sea_temperature_diff_thresh"] ) 
        else:
            raise ValueError("ta must be in ds and lsm must be provided for land sea temperature filter")   
    
    #Filter objects based on local temperature decrease
    if mask_options.filters["temperature_change_filter"]:
        if ("t_change" in list(ds.keys())):
            #Calculate the mean temperature change over each labelled region
            mean_t_change = ds["t_change"].groupby(labels_da.rename("label")).mean().to_series()
            props_df["mean_t_change"] = mean_t_change
            cond = cond & ( mean_t_change < mask_options.thresholds["temperature_change_thresh"] ) 
        else:
            raise ValueError("t_change must be provided in ds for temperature change filter")
        
    #Filter objects based on local humidity increase
    if mask_options.filters["humidity_change_filter"]:
        if ("q_change" in list(ds.keys())):
            #Calculate the mean humidity change over each labelled region
            mean_q_change = ds["q_change"].groupby(labels_da.rename("label")).mean().to_series()
            props_df["mean_q_change"] = mean_q_change
            cond = cond & ( mean_q_change > mask_options.thresholds["humidity_change_thresh"] ) 
        else:
            raise ValueError("q_change must be provided in ds for humidity change filter")
        
    #Filter objects based on local onshore wind speed increase
    if mask_options.filters["wind_change_filter"]:
        if ("wind_change" in list(ds.keys())):
            #Calculate the mean wind change over each labelled region
            mean_wind_change = ds["wind_change"].groupby(labels_da.rename("label")).mean().to_series()
            props_df["mean_wind_change"] = mean_wind_change
            cond = cond & ( mean_wind_change > mask_options.thresholds["wind_change_thresh"] ) 
        else:
            raise ValueError("wind_change must be provided in ds for wind change filter")

    #Subset label dataframe based on these conditions
    props_df = props_df.loc[cond]
    props_df["time_utc"] = time

    #Remove labels that don't meet conditions from mask
    mask = xr.where(labels_da.isin(props_df.loc[cond].index),1,0)

    #Assign attributes to mask
    mask = mask.assign_attrs({
        "filters":mask_options.filters,
        "thresholds":mask_options.thresholds,
        "other_options":mask_options.options}).astype(bool)

    #Combine mask with label array and props_df
    ds = xr.merge([
        mask.assign_coords({"time":time}).expand_dims("time").rename("mask"),
        xr.where(
            labels_da.isin(props_df.index),labels_da,0
            ).astype(np.int16).assign_coords({"time":time}).expand_dims("time").rename("filtered_labels"),
        labels_da.astype(np.int16).assign_coords({"time":time}).expand_dims("time").rename("all_labels")
        ])
        #xr.Dataset.from_dataframe(props_df).assign_coords({"time":time}).expand_dims("time"),

    if (output_land_sea_temperature_diff) & (mask_options.filters["land_sea_temperature_filter"]):
        ds = xr.merge((ds,land_sea_temp_diff.expand_dims("time").rename("land_sea_temp_diff")))

    #Save props_df to csv and output with label index as a column
    if props_df_output_path is None:
        props_df_output_path = "/scratch/gb02/ab4502/tmp/props_df.csv"
    props_df = props_df.reset_index().rename({"index":"label"})
    props_df.to_csv(props_df_output_path,mode="a",header=False,index=False)

    return ds, props_df

def land_sea_temperature_diff_rolling_max(ts,lsm,R_km,dy,dx):

    """
    From a dataarray of surface temperature (ts), calculate the land-sea temperature difference for each point in the domain.
    For each point, the land temperature is filtered with a rolling maximum over a radius of R km and the sea temperature is filtered
    with a rolling mean. Then for each point, the land sea temperature difference is calculated as the difference between the closest land and closest sea point.

    ## Input
    * ts: xarray dataarray of surface temperatures
    * lsm: xarray dataarray of the land-sea mask
    * R_km: target radius of the rolling maximum filter in km. note that actual radius is calculated by dividing this number by the the mean grid spacing in x and y
    * dy: y grid spacing in km
    * dx: x grid spacing in km
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

    #Now look up the N closest land/sea points and store in an extra third dimension "points"
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
    print(f"INFO: Radius for land_sea_temperature_difference smoothing in longitude direction ({radius_x} pixels) will vary with latitude, between {radius_x * np.min(dx.data)} km and {radius_x * np.max(dx.data)} km.\nINFO: Radius in latitude direction is {radius_y * np.mean(dy.data)} km ({radius_y} pixels)")
    
    land_ts = xr.DataArray(
        scipy.ndimage.generic_filter(ts.where(lsm==1, np.nan), np.nanmax, footprint=footprint),
        coords=ts.coords, dims=ts.dims).where(lsm==1, np.nan)
    sea_ts = xr.DataArray(
        scipy.ndimage.generic_filter(ts.where(lsm==0, np.nan), np.nanmean, footprint=footprint),
        coords=ts.coords, dims=ts.dims).where(lsm==0, np.nan)

    #Calculate land sea difference
    land_minus_sea = (
        land_ts.sel(lon=target_lon_land,lat=target_lat_land).assign_coords({"lat":lsm.lat,"lon":lsm.lon}) - \
            sea_ts.sel(lon=target_lon_sea,lat=target_lat_sea).assign_coords({"lat":lsm.lat,"lon":lsm.lon})
            )

    return land_minus_sea


def land_sea_temperature_diff(ts,lsm,N,weighting="none",sigma=0.1):

    """
    From a dataarray of surface temperature (ts), calculate the land-sea temperature difference for each point in the domain.
    For each point, the land temperature is taken as the average of the closest N land points, and the sea temperature is taken as the N closest sea points. There is an option to weight those points before averaging using a gaussian function.

    ## Input
    * ts: xarray dataarray of surface temperatures
    * lsm: xarray dataarray of the land-sea mask
    * N: number of nearest neighbour points to use for defining land and sea temperatures for each point
    * weighting: can be "none" to use equal weights when doing the nearest neighbour average, or "gaussian" to use an inverse distance gaussian function
    * sigma: standard deviation of the inverse distance weighting function when averaging nearest points, if weighting="gaussian"

    TODO: Nearest neighbour max? If temperature has dropped locally due to sea breeze front, then the land temperature could become colder than the ocean temperature. Or use previous time for the calculation?
    TODO: Use pyproj for gaussian weighting function.
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

    #Now look up the N closest land/sea points and store in an extra third dimension "points"
    _,land_ind = land_kdt.query(np.array([yy.flatten(),xx.flatten()]).T, N)
    target_lon_land = land_lon[land_ind.reshape((lat.shape[0],lon.shape[0],-1))]
    target_lat_land = land_lat[land_ind.reshape((lat.shape[0],lon.shape[0],-1))]
    target_lon_land = xr.DataArray(target_lon_land,dims=("lat","lon","points"),coords={"lat":lsm.lat,"lon":lsm.lon,"points":np.arange(N)})
    target_lat_land = xr.DataArray(target_lat_land,dims=("lat","lon","points"),coords={"lat":lsm.lat,"lon":lsm.lon,"points":np.arange(N)})

    #For sea NN
    _,sea_ind = sea_kdt.query(np.array([yy.flatten(),xx.flatten()]).T, N)
    target_lon_sea = sea_lon[sea_ind.reshape((lat.shape[0],lon.shape[0],-1))]
    target_lat_sea = sea_lat[sea_ind.reshape((lat.shape[0],lon.shape[0],-1))]
    target_lon_sea = xr.DataArray(target_lon_sea,dims=("lat","lon","points"),coords={"lat":lsm.lat,"lon":lsm.lon,"points":np.arange(N)})
    target_lat_sea = xr.DataArray(target_lat_sea,dims=("lat","lon","points"),coords={"lat":lsm.lat,"lon":lsm.lon,"points":np.arange(N)})

    if weighting=="none":
        #If no weighting, just take the average of neighbouring N land/ocean points
        land_minus_sea = (
            ts.sel(lon=target_lon_land,lat=target_lat_land).mean("points").assign_coords({"lat":lsm.lat,"lon":lsm.lon}) - \
                ts.sel(lon=target_lon_sea,lat=target_lat_sea).mean("points").assign_coords({"lat":lsm.lat,"lon":lsm.lon})
                )
    elif weighting=="max":
        #If max, then take the neighbourhood maximum over the land, and mean over the ocean
        land_minus_sea = (
            ts.sel(lon=target_lon_land,lat=target_lat_land).max("points").assign_coords({"lat":lsm.lat,"lon":lsm.lon}) - \
                ts.sel(lon=target_lon_sea,lat=target_lat_sea).mean("points").assign_coords({"lat":lsm.lat,"lon":lsm.lon})
                )        
    elif weighting=="gaussian":
        #Define weighting functions as e^[ -( x**2 / 2*sigma**2 ) ) ], where for each point, x is the distance between the cloeset
        # land/ocean point and all other neighbouring land/ocean points

        #For land points...
        #Get the distance between each point and closest N neighbouring land points
        land_dist_deg = (np.sqrt(((target_lon_land - target_lon_land.mean("points"))**2 + \
                (target_lat_land - target_lat_land.mean("points"))**2)))

        #For each point find the closest land point in lat/lon coordinates
        closest_land_lon = target_lon_land.isel(points=land_dist_deg.idxmin("points").astype(int))
        closest_land_lat = target_lat_land.isel(points=land_dist_deg.idxmin("points").astype(int))

        #Create the weighting function for land points
        x_land = (np.sqrt(((target_lon_land - closest_land_lon)**2 + \
                (target_lat_land - closest_land_lat)**2)))
        weights_land = np.exp(-(x_land**2/(2*sigma**2)))

        #Do the same for the sea points
        sea_dist_deg = (np.sqrt(((target_lon_sea - target_lon_sea.mean("points"))**2 + \
                (target_lat_sea - target_lat_sea.mean("points"))**2)))
        closest_sea_lon = target_lon_sea.isel(points=sea_dist_deg.idxmin("points").astype(int))
        closest_sea_lat = target_lat_sea.isel(points=sea_dist_deg.idxmin("points").astype(int))
        x_sea = (np.sqrt(((target_lon_sea - closest_sea_lon)**2 + \
                (target_lat_sea - closest_sea_lat)**2)))
        weights_sea = np.exp(-(x_sea**2/(2*sigma**2)))

        #Calculate the land minus sea temperatures
        land_minus_sea = (
            (ts.sel(lon=target_lon_land,lat=target_lat_land) * weights_land).sum("points") / weights_land.sum("points") - \
            (ts.sel(lon=target_lon_sea,lat=target_lat_sea) * weights_sea).sum("points") / weights_sea.sum("points")
            )

    else:

        raise ValueError("weighting must equal ''none'', ''max'' or ''gaussian''")

    return land_minus_sea

def land_sea_temperature_grad(ts,lsm,N,angle_ds,weighting="none",sigma=0.1):

    """
    Calculate the land sea temperature difference using land_sea_temperature_diff(), and then convert to a gradient by dividing by distance from the coast

    ## Input
    * ts: xarray dataarray of surface temperatures
    * lsm: xarray dataarray of the land-sea mask
    * N: number of nearest neighbour points to use for defining land and sea temperatures for each point
    * angle_ds: dataset of coastline angles created by get_coastline_angles.py. contains coastline mask that is used here
    * dx: grid spacing in km. 
    * weighting: can be "none" to use equal weights when doing the nearest neighbour average, or "gaussian" to use an inverse distance gaussian function
    * sigma: standard deviation of the inverse distance weighting function when averaging nearest points, if weighting="gaussian"
    """

    #Calculate the land-sea temperature contrast
    land_minus_sea = land_sea_temperature_diff(ts,lsm,N,weighting=weighting,sigma=sigma)

    #Get approximate grid spacing by using the mean spacing in x and y over the whole domain
    lat = lsm.lat.values
    lon = lsm.lon.values
    xx,yy = np.meshgrid(lon,lat)
    dx,dy = mpcalc.lat_lon_grid_deltas(xx,yy)
    dx = ((dy.mean() + dx.mean())/2).to("km").magnitude

    #Calculate the distance between each spatial point and the closest coastline
    dist = np.zeros(xx.shape)
    x_coast = xx[angle_ds.coast==1]
    y_coast = yy[angle_ds.coast==1]
    for i in tqdm.tqdm(np.arange(xx.shape[0])):
        for j in np.arange(xx.shape[1]):
            dist[i,j] = np.min(latlon_dist(yy[i,j],xx[i,j],y_coast,x_coast))

    #Divide the land sea temperature contrast by the distance from the coast to give a kind-of gradient
    #If the point lies on the coastline, then the temperature is divided by the grid spacing to give a gradient
    land_minus_sea_grad = xr.where(
        dist==0,
        land_minus_sea / dx,
        land_minus_sea / dist)

    return land_minus_sea_grad    

def filter_ds_driver(field,p=95,hourly_change_ds=None,ta=None,lsm=None,angle_ds=None,save_mask=False,filter_out_path=None,props_df_out_path=None,skipna=False,**kwargs):

    """
    Driver function for filter_ds that creates a binary mask from a sea breeze diagnostic field, and works with a time dimension using xr.map_blocks. Docstring copied and adapted from filter_ds:

    ## Description
    Take a sea breeze diagnostic field, then create a binary mask, then filter that mask for sea breeze objects based on several conditions related to those objects. Works for a 3d lat/lon/time field as an xarray dataset. The binary mask is created by taking exceedences of a percentile value as calculated from 'field'.

    ## Input
    ### Required
    * field: xarray dataarray of a sea breeze diagnostic, with dims lat/lon/time. See sea_breeze.sea_breeze_funcs for diagnostic functions, as well as sea_breeze.sea_breeze_filters.fuzzy_function_combine.

    ### Optional
    * p: the percentile used to create a binary mask from field

    * hourly_change_ds: an xarray dataset containing "t_change" (temperature change), "q_change" (specific humidity change) and "wind_change" (onshore wind speed change) as variables (as output from sea_breeze_funcs.hourly_change). Needed for temperature_change_filter/humidity_change_filter/wind_change_filter

    * ta: xarray dataarray of surface temperature, with the same coords and dims as field. Needed for land_sea_temperature_filter

    * lsm: xarray dataarray of the land-sea mask. Needed for land_sea_temperature_filter

    * angle_ds: xarray dataset of coastline angles created by sea_breeze.load_model_data.get_coastline_angles_kernel. Needed for orientation_filter, dist_to_coast_filter.

    * save_mask: whether to save the output mask dataset to disk

    * filter_out_path: if save_mask=True, then where to save the filtered mask output

    * props_df_output_path: path to output a csv file of the properties of each object    

    * skip_na: if the field contains nans, then the calculation of the field percentile will ignore nans if this is set to true. Uses the climtas package.

    * **kwargs: options for filtering the sea breeze objects passed to filter_ds (see below)

    ## Options (kwargs)

    ### Filters
    * orientation_filter: Filter candidate objects based on orientation parallel to the coastline True/False (default False). Requires angle_ds to be provided

    * aspect_filter: Filter candidate objects based on aspect ratio True/False (default True)

    * area_filter: Filter candidate objects based on area True/False (default True)

    * dist_to_coast_filter: Filter candidate objects based on distance to the coast True/False (default False). Requires angle_ds to be provided

    * time_filter: Filter candidate objects based on local solar time True/False (default True)

    * land_sea_temperature_filter: Filter candidate objects based on land sea temperature difference True/False (default False). Requires ta and lsm to be provided

    * temperature_change_filter: Filter candidate objects based on mean local temperature decrease True/False (default False). Requires t_change to be provided as a variable in hourly_change_ds.

    * humidity_change_filter: Filter candidate objects based on mean local humidity increase True/False (default False). Requires q_change to be provided as a variable in hourly_change_ds.

    * wind_change_filter: Filter candidate objects based on mean local onshore wind speed increase True/False (default False). Requires wind_change to be provided as a variable in hourly_change_ds.

    * output_land_sea_temperature_diff: output the land sea temperature difference as a dataarray (bool, default False)        

    ### Thresholds
    * orientation_tol: Tolerance in degrees for orientation filter (default 30 degrees)
    * area_thresh: Minimum area in km^2 for area filter if area_filter_units='kms' (defailt 50 km^2)
    * area_thresh_pixels: Minimum area in pixels for area filter if area_filter_units='pixels' (defailt 20)
    * aspect_thresh: Minimum aspect ratio for aspect filter (default 3)
    * hour_min_lst: Minimum local solar time in hours for time filter (default 9)
    * hour_max_lst: Maximum local solar time in hours for time filter (default 22)
    * land_sea_temperature_diff_thresh: Minimum land sea temperature difference for land sea temperature filter (default 0)
    * distance_to_coast_thresh: Maximum distance to coast in km for distance to coast filter (default 300 km)
    * temperature_change_thresh: Maximum temperature change for temperature change filter, same units as t_change (default 0)
    * humidity_change_thresh: Minimum humidity change for humidity change filter, same units as q_change (default 0)
    * wind_change_thresh: Minimum wind change for onshore wind change filter, same units as wind_change (default 0)

    ### Other options
    * land_sea_temperature_radius: Radius of the rolling maximum/mean filter for land sea temperature difference, in km (default 50 km). NOTE that this radius is converted to pixels by using the mean grid spacing in x and y, and will therefore vary with latitude.
    * area_filter_units: Is area filtered by pixels or kms? Either 'pixels' or 'kms'

    ## Output
    * mask: binary mask of sea breeze objects after filtering

    TODO Propagation speed or persistance? Note this is not straightforward. Will need to be in a separate function with input having a time dimension.

    """

    if save_mask:
        if filter_out_path is None:
            raise ValueError("filter_out_path must be provided if save_mask is True")

    #Load sea breeze function from sea_breeze_funcs
    field = field.chunk({"time":1,"lat":-1,"lon":-1})

    #Mask based on percentile values
    thresh = percentile(field,p=p,skipna=skipna).compute()
    print("Using threshold: ",str(thresh))
    ds = binary_mask(field, thresh)

    #If angle_ds is provided, add to kwargs to pass to filtering
    if angle_ds is not None:
        kwargs["angle_ds"] = angle_ds.compute()

    #If lsm is provided, add to kwargs to pass to filtering
    if lsm is not None:
        kwargs["lsm"] = lsm.compute()
    
    #If ta is provided, re-chunk and combine with the mask dataset
    if ta is None:
        pass
    else:
        ta=ta.chunk({"time":1,"lat":-1,"lon":-1})
        ds = xr.merge((ds,ta.rename("ta")))   

    #If we have given a path to hourly change data, load it and combine with the mask dataset
    if hourly_change_ds is None:
        pass
    else:
        hourly_change_ds =hourly_change_ds.chunk({"time":1,"lat":-1,"lon":-1})
        ds = xr.merge((ds,hourly_change_ds))         

    #We will apply the filtering using map_blocks. So first, need create a "template" from the first time step
    if props_df_out_path is None:
        props_df_out_path = "/scratch/gb02/ab4502/tmp/props_df.csv"
    template,props_df_template = filter_ds(ds.isel(time=0), **kwargs)
    template = template.chunk({"time":1}).reindex({"time":ds.time},fill_value=False).chunk({"time":1})

    #Setup the output dafaframe for saving sea breeze object properties as csv files
    initialise_props_df_output(props_df_out_path,props_df_template)

    #Apply the filtering
    filtered_mask = ds.map_blocks(
        process_time_slice,        
        template=template,
        kwargs=kwargs)

    #Set some extra attributes
    filtered_mask["mask"] = filtered_mask["mask"].assign_attrs({"threshold":ds.mask.threshold})
    
    #Save the filtered mask if required
    if save_mask:
        filtered_mask.to_netcdf(filter_out_path)
    
    return filtered_mask

def filter_ds_driver_from_disk(field_path,field_name,lat_slice,lon_slice,t1=None,t2=None,hourly_change_path=None,ta=None,lsm=None,angle_ds=None,save_mask=False,filter_out_path=None,props_df_out_path=None,skipna=False,**kwargs):

    """
    Driver function for filter_ds, loading data from disk
    """

    if save_mask:
        if filter_out_path is None:
            raise ValueError("filter_out_path must be provided if save_mask is True")

    #Load sea breeze function from sea_breeze_funcs
    field = xr.open_dataset(field_path,chunks={"time":1,"lat":-1,"lon":-1})[field_name].sel(lat=lat_slice,lon=lon_slice)
    # lat_slice = slice(field.lat.min().values,field.lat.max().values)
    # lon_slice = slice(field.lon.min().values,field.lon.max().values)
    if (t1 is not None) & (t2 is not None):
        field = field.sel(time=slice(t1,t2))
    else:
        t1 = field.time.min().values
        t2 = field.time.max().values

    #Mask based on percentile values
    thresh = percentile(field,95,skipna=False)
    print("Using threshold: ",str(thresh))
    ds = binary_mask(field, thresh)

    #If angle_ds is provided, add to kwargs to pass to filtering
    if angle_ds is not None:
        kwargs["angle_ds"] = angle_ds.compute()

    #If lsm is provided, add to kwargs to pass to filtering
    if lsm is not None:
        kwargs["lsm"] = lsm.compute()
    
    #If ta is provided, re-chunk and combine with the mask dataset
    if ta is None:
        pass
    else:
        ta=ta.chunk({"time":1,"lat":-1,"lon":-1})
        ds = xr.merge((ds,ta.rename("ta")))   

    #If we have given a path to hourly change data, load it and combine with the mask dataset
    if hourly_change_path is None:
        pass
    else:
        hourly_change = xr.open_dataset(
            hourly_change_path,chunks={"time":1,"lat":-1,"lon":-1}).sel(time=slice(t1,t2),lat=lat_slice,lon=lon_slice)
        ds = xr.merge((ds,hourly_change))         

    #We will apply the filtering using map_blocks. So first, need create a "template" from the first time step
    if props_df_out_path is None:
        props_df_out_path = "/scratch/gb02/ab4502/tmp/props_df.csv"
    template,props_df_template = filter_ds(ds.isel(time=0), kwargs)
    template = template.chunk({"time":1}).reindex({"time":ds.time},fill_value=0).chunk({"time":1})

    #Setup the output dafaframe for saving sea breeze object properties as csv files
    initialise_props_df_output(props_df_out_path,props_df_template)

    #Apply the filtering
    filtered_mask = ds.map_blocks(
        process_time_slice,        
        template=template,
        kwargs=kwargs)
    
    #Save the filtered mask if required
    if save_mask:
        filtered_mask.to_netcdf(filter_out_path)
    
    return filtered_mask
