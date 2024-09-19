import numpy as np
import xarray as xr
import tqdm
from skimage.segmentation import flood
import scipy

def latlon_dist(lat, lon, lats, lons):

        #Calculate great circle distance (Harversine) between a lat lon point (lat, lon) and a list of lat lon
        # points (lats, lons)

        R = 6373.0

        lat1 = np.deg2rad(lat)
        lon1 = np.deg2rad(lon)
        lat2 = np.deg2rad(lats)
        lon2 = np.deg2rad(lons)

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        return (R * c)

def get_coastline(lsm):

    '''
    From a binary land-sea mask (lsm), define a coastline using gradient in the lsm
    '''
    coastline = (np.abs(((lsm.diff("lat")) + (lsm.diff("lon")))) >= 1)
    lsm_ds = xr.Dataset({"lsm":lsm,"coast":coastline})

    return lsm_ds

def get_coastline_angle(lsm_ds, r=100, R=300, N=None, small_coast_thresh=0.001):

    '''
    From a coastline mask, define the local angle at each point.
    This is done by looping over each point on the coastline, and fitting a linear polynomial
    using all coastline points within r km.
    Note also that this function ignores small coastline segments away from the mainland, 
    and removes such coasts where the proportion of connected coastline pixels is less than
    some fraction of the total domain (small_coast_thresh). The updated coastline is added
    to the lsm_ds dataset.

    After local coasline angles are defined, this function also defines coastline angles for all
    points in the domain, by taking the circular average within R km, or the N closest points
    The coastline binary mask is also expanded by R km.

    The angles output is in degrees from north, ranging from 0 to 180 degrees
    '''

    assert (R is not None) | (N is not None), "R or N must be an integer"

    #Define the x and y grid, and remove the xarray metadata for the coastline dataset
    x = lsm_ds.lon.values
    y = lsm_ds.lat.values
    xx,yy = np.meshgrid(x,y)
    coastline = lsm_ds.coast.values

    #Initialise an "angle" array
    angles = np.ones(coastline.shape) * np.nan

    #Loop over all points in the spatial domain
    print("Defining coastline angles...")
    for xi in tqdm.tqdm(range(coastline.shape[1])):
        for yi in range(coastline.shape[0]):

            #If this point is a coastline...
            if coastline[yi,xi] == 1:

                #Get all connected points using scikit image
                connected = flood(coastline,(yi,xi)) 

                #Now, if the number of connected points is greater than some threshold 
                #(here, 0.001 would represent 0.1% of points of the whole domain), do the following
                #This is used to exclude small islands
                if connected.sum() / (connected.shape[0] * connected.shape[1]) > small_coast_thresh:

                    #Mask the connected points that are further away than r km
                    dist_mask = (latlon_dist(y[yi], x[xi], yy, xx) <= r) * 1
                    connected_close = connected * dist_mask
                    
                    #Fit a linear polynomial through the neighbouring connected points, to estimate the local orientation of the coast
                    y_ind, x_ind  = np.where(connected_close)
                    fit_xy = np.polyfit(x[x_ind],y[y_ind],deg=1)
                    yf = np.polyval(fit_xy, x)

                    #From this polynomial, calculate the angle from north
                    angle_fit = np.arctan2(yf[-1] - yf[0], x[-1] - x[0])
                    angle_fit = np.rad2deg(angle_fit)
                    angle_fit = -(angle_fit-90) % 360

                    #Keep track of all the angles
                    angles[yi,xi] = angle_fit
    
    #Update dataset
    lsm_ds["angles"] = (("lat","lon"),angles)
    lsm_ds["coastline_main"] = (("lat","lon"), ~np.isnan(angles))

    #Now loop over all points in the domain, and find coastlines close by
    #Take the average of coastline angles using a circular average
    closest_coast_angle = np.zeros(xx.shape)
    if N is not None:
        print("Expanding angles to average of closest "+str(N)+" points within "+str(R)+" kms...")
        flattened_angles = lsm_ds.angles.values.flatten()
    else:
        print("Expanding angles to average of points within "+str(R)+" kms...")
    for xi in tqdm.tqdm(range(xx.shape[0])):
        for yi in range(xx.shape[1]):
            dist = latlon_dist(yy[xi,yi], xx[xi,yi], yy, xx)
            if N is not None:
                #If N is an integer, then points are assigned a coastline angle using
                #the average of the N closest coastline points within R km
                closest_coast_angle[xi,yi] = np.rad2deg(
                    scipy.stats.circmean(
                    np.deg2rad(
                        flattened_angles[
                            np.argsort(xr.where((lsm_ds.coastline_main) & (dist<=R),dist,np.nan).values,axis=None)[0:N]
                            ]), 
                    high=np.pi, low=0, nan_policy="omit"))
            else:
                #If N is set to None, a radius of R km is used to define the coastline
                #angle for each point
                closest_coast_angle[xi,yi] = np.rad2deg(
                    scipy.stats.circmean(
                    np.deg2rad(
                        xr.where((lsm_ds.coastline_main) & (dist<=R),lsm_ds.angles,np.nan)
                        ), 
                    high=np.pi, low=0, nan_policy="omit"))

    #Update dataset
    lsm_ds["angles_expanded"] = (("lat","lon"),closest_coast_angle)
    lsm_ds["coastline_expanded"] = (("lat","lon"), ~np.isnan(closest_coast_angle))

    return lsm_ds