import scipy.ndimage
import skimage
import scipy
import tqdm
import numpy as np
import pandas as pd
import xarray as xr
import datetime as dt

def binary_mask(field,thresh):

    """
    Take a field that diagnoses sea breeze potential and create a binary sea breeze mask based on a threshold

    Absolute versus percentile/perturbation thresholds? Standard deviation threshold
    """

    return (field>thresh)*1

class Mask_Options:
    
    #Options.
    # TODO move to function input and save options as class
    # TODO: Change area to area units instead of pixels. Can use spacing arg in regionprops

    def __init__(self):
        self.filters = {
            "orientation_filter":True,
            "aspect_filter":True,
            "area_filter":True,
            "dist_to_coast_filter":True,
            "time_filter":True,
            "land_sea_temperature_filter":True
        }

        self.thresholds = {
            "orientation_tol":30,                     #Within 30 degrees of coastline
            "area_thresh":10,                         #Area of 10 pixels or greater
            "aspect_thresh":3,                        #Aspect ratio of 3 or greater
            "time_thresh_lst":12,                     #LST hour must be greater than 12 (and therefore less than or equal to 23),
            "land_sea_temperature_diff_thresh":0,     #Land sea temperature difference must be greater than zero
            "distance_to_coast_thresh":300}           #Within 300 km of the coast

    def set_options(self,kwargs):
        for k, val in kwargs.items():
            if k in self.filters.keys():
                self.filters[k] = val
            elif k in self.thresholds.keys():
                self.thresholds[k] = val
            else:
                raise ValueError(f"{k} not a valid option")
        return self

    def __str__(self):
        return f"Filters: {self.filters} \n Thresholds: {self.thresholds}"

def filter_sea_breeze(mask,angle_ds,ta,time,lsm,resolution_km,**kwargs):
    
    """

    ## Input
    * mask: binary mask of sea breeze objects
    * options: dictionary of options for filtering sea breeze objects
    * angle_ds: xarray dataset of coastline angles created by get_coastline_angles.py
    * ta: xarray dataarray of surface temperatures
    * time: time of the data in the form %Y-%m-%d %H:%M
    * lsm: xarray dataarray of the land-sea mask
    * resolution_km: resolution of the grid in km

    ## Options
    * TODO 

    ## Output
    * mask: binary mask of sea breeze objects after filtering

    Take a binary sea breeze mask and identify objects, then filter it for sea breezes based on several conditions related to those objects

    - Area (number of pixels) (done)
    - Eccentricity or aspect ratio (done)
    - Orientation of object relative to the coastline (done)
    - Within some distance to the coast (done)
    - Positive land-sea temperature contrast/gradient (done)
    - Local time (done - criteria is that the local time is between 12:00 and 23:59 local solar time)
    - TODO Local increase/decrease in humdity/wind speed/temperature?
    - TODO Fuzzy method? Separate function.
    - TODO Save properties of each object for further analysis
    - TODO Propagation speed or persistance? Note this is not straightforward. Will need to be in a separate function with input having a time dimension.

    """

    #Set options
    mask_options = Mask_Options().set_options(kwargs)

    #From a binary (mask) array of candidate sea breeze objects, label from 1 to N
    labels = skimage.measure.label(mask)

    #Using skimage, return properties for each candidate object
    region_props = skimage.measure.regionprops(labels,spacing=(1,1))

    #Get longitudes of image for the purpose of converting to local solar time
    lons = angle_ds.lon.values

    #For each object create lists of relevant properties
    labs = np.array([region_props[i].label for i in np.arange(len(region_props))])                           #Object label number
    eccen = [region_props[i].eccentricity for i in np.arange(len(region_props))]                             #Eccentricity
    area = [region_props[i].area for i in np.arange(len(region_props))]                                      #Area
    orient = np.rad2deg(np.array([region_props[i].orientation for i in np.arange(len(region_props))]))       #Orientation angle
    major = np.rad2deg(np.array([region_props[i].axis_major_length for i in np.arange(len(region_props))]))  #Major axis length
    minor = np.rad2deg(np.array([region_props[i].axis_minor_length for i in np.arange(len(region_props))]))  #Major axis length
    centroid_lons = [lons[np.round(region_props[i].centroid[1]).astype(int)] for i in np.arange(len(region_props))]  
    lst = [pd.to_datetime(time) + dt.timedelta(hours=l / 180 * 12) for l in centroid_lons]                   #Local solar time

    #Create pandas dataframe with object properties, index by label number
    props_df = pd.DataFrame({
        "eccen":eccen,
        "area":area,
        "orient":orient,
        "major":major,
        "minor":minor,
        "lst":lst}, index=labs)        
    props_df["aspect"] = props_df.major/props_df.minor

    #Calculate the mean angle of nearby coastlines over each labelled region using xarray apply ufunc and scipy circmean
    def circmean_wrapper(data):
        return scipy.stats.circmean(data, low=-90, high=90)
    labels_da = xr.DataArray(labels, dims=angle_ds.dims, coords=angle_ds.coords)
    mean_angles = angle_ds.angle_interp.groupby(labels_da.rename("label")).map(
            lambda x: xr.apply_ufunc(
                circmean_wrapper,
                x,
                input_core_dims=[['stacked_lat_lon']],
                vectorize=True
            )
        ).to_series()    
    
    #Calculate the mean distance to coastline over each labelled region
    mean_dist = angle_ds.min_coast_dist.groupby(labels_da.rename("label")).mean().to_series()

    #Calculate the local (smoothed) difference between nearest land and sea temperatures
    #Here use a radius of 50 km for the smoothing (radial max over land, mean over ocean)
    land_sea_temp_diff = land_sea_temperature_diff_rolling_max(ta,lsm,R_km=50,resolution_km=resolution_km)

    #Calculate the mean land-sea temperature difference over each labelled region
    mean_land_sea_temp_diff = land_sea_temp_diff.groupby(labels_da.rename("label")).mean().to_series()

    #Create condition to keep objects
    cond = ( (props_df.orient - mean_angles).abs() <= mask_options.thresholds["orientation_tol"]) &\
            ( props_df.aspect >= mask_options.thresholds["aspect_thresh"] ) &\
            ( props_df.area >= mask_options.thresholds["area_thresh"] ) &\
            ( mean_dist <= mask_options.thresholds["distance_to_coast_thresh"] ) &\
            ( props_df.lst.dt.hour >= mask_options.thresholds["time_thresh_lst"] ) &\
            ( mean_land_sea_temp_diff > mask_options.thresholds["land_sea_temperature_diff_thresh"] )
    
    #Subset label dataframe based on these conditions
    props_df = props_df.loc[cond]

    #Remove labels that don't meet conditions from mask
    mask = xr.where(labels_da.isin(props_df.loc[cond].index),1,0)

    #Assign attributes to mask
    mask = mask.assign_attrs({
        "filters":mask_options.filters,
        "thresholds":mask_options.thresholds,
        "resolution_km":resolution_km})    

    #Combine mask with label array and props_df
    ds = xr.merge([
        xr.Dataset.from_dataframe(props_df),
        mask.assign_coords({"time":time}).expand_dims("time").rename("mask"),
        xr.where(labels_da.isin(props_df.index),labels_da,0).assign_coords({"time":time}).expand_dims("time").rename("label")
        ])

    return ds

def land_sea_temperature_diff_rolling_max(ts,lsm,R_km,resolution_km):

    """
    From a dataarray of surface temperature (ts), calculate the land-sea temperature difference for each point in the domain.
    For each point, the land temperature is filtered with a rolling maximum over a radius of R km and the sea temperature is filtered
    with a rolling mean. Then for each point, the land sea temperature difference is calculated as the difference between the closest land and closest sea point.

    ## Input
    * ts: xarray dataarray of surface temperatures
    * lsm: xarray dataarray of the land-sea mask
    * R: radius of the rolling maximum filter in km
    * resolution_km: resolution of the grid in km

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
    def create_circular_footprint(radius):
        """Create a circular footprint with a given radius and resolution."""
        y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
        footprint = x**2 + y**2 <= radius**2
        return footprint    
    radius = int(R_km / resolution_km)
    footprint = create_circular_footprint(radius)    
    
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