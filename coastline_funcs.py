import numpy as np
import xarray as xr
import tqdm
import scipy
from skimage.segmentation import find_boundaries, flood
import skimage.measure as measure
import skimage.morphology as morphology

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

def get_coastline_angle_sorting(lsm):

    #Get coastline and labelled landmass masks using skimage
    coastline_masks, label_masks = label_and_find_coast(lsm)
    lon = lsm.lon.values
    lat = lsm.lat.values

    #Sort coastline indices in a counter-clockwise direction
    sorted_inds_ls, bound_xind_ls, bound_yind_ls  = order_coastline_points(coastline_masks, label_masks)

    #Loop over each labelled land mass in the dictionary, and find angles
    #NOTE currently just finding the angle between every point and fifth-most next point.
    #Would rather take the circular average of points 2-N
    labels = list(label_masks.keys())
    angles = np.zeros(lsm.shape) * np.nan
    for l in labels:
        bound_xind = bound_xind_ls[l]
        bound_yind = bound_yind_ls[l]
        sorted_inds = sorted_inds_ls[l]
        for ind in range(len(sorted_inds)-5):
            angles[bound_xind[sorted_inds[ind]], 
                bound_yind[sorted_inds[ind]]] = np.arctan2(
                    lat[bound_xind[sorted_inds[ind+5]]] - lat[bound_xind[sorted_inds[ind]]],
                    lon[bound_yind[sorted_inds[ind+5]]] - lon[bound_yind[sorted_inds[ind]]])

    angles = np.rad2deg(angles)
    angles = -(angles-90) % 360

    return angles

def label_and_find_coast(lsm,erosion_footprint=morphology.disk(2),size_thresh=10):

    '''
    Given a binary land-sea mask Dataarray, label the array and find boundaries (coastline)
    using skimage

    Inputs:
    erosion_footprint: A disk used to smooth/erode the binary lsm, see:
    https://scikit-image.org/docs/stable/auto_examples/applications/plot_morphology.html#erosion
    size_thresh: A threshold on the number of pixels for a land mass to be considered as a coastline.
    note this is applied AFTER the erosion 

    Output:
    coastline_masks: A dictionary with keys as label numbers from 0 to N, and items as a coastline mask
    label_masks: A dictionary with keys as label numbers from 0 to N, and items as a label mask
    '''

    #Label the binary land sea mask, while applying an erosion method to remove
    #parts of the coast connected by skinny land bridges (the algorithm for ordering 
    #the coastline points has trouble with that)
    labelled_lsm = measure.label(morphology.erosion(lsm,footprint=erosion_footprint),connectivity=1)

    #Remove labelled land masses smaller than some size threshold
    unique_labels = np.unique(labelled_lsm)
    for l in unique_labels:
        if l>0:
            if (labelled_lsm==l).sum() <= size_thresh:
                labelled_lsm[labelled_lsm==l] = 0
    unique_labels = np.unique(labelled_lsm)                

    #Get boundaries (coastlines)
    coastline_masks = []
    label_masks = []
    for l in unique_labels:
        if l>0:
            coastline_masks.append(find_boundaries(labelled_lsm==l, connectivity=1, mode="inner"))
            label_masks.append(labelled_lsm==l)

    coastline_masks = dict(zip(np.arange(len(unique_labels)),coastline_masks))
    label_masks = dict(zip(np.arange(len(unique_labels)),label_masks))
    return coastline_masks, label_masks

def order_coastline_points(coastline_masks, label_masks):

    '''
    For a set of coastline masks, this function uses an algorithm to order the coastline points in 
    a counter-clockwise direction around the centre of the land mass

    Ouputs are dictionaries of sorted coast indices, as well as the standard (x-y) sorted x inds and y inds
    '''
    
    labels = list(label_masks.keys())

    #Loop over each labelled land mass in the dictionary
    sorted_inds_ls = []
    bound_xind_ls = []
    bound_yind_ls = []
    for l in labels:

        #Get the centroid of the land mass
        centre_xind, centre_yind = measure.centroid(label_masks[l]).round().astype(int)

        #Get the x,y indices of the coasline from the mask
        bound_xind, bound_yind= np.where(coastline_masks[l]==1)

        #Get a list of indices from 0 to N of coastline points
        point_inds = np.arange(len(bound_xind))

        #Initialise the (counter-clockwise) sorted list of coastline points, and the initial point (a)
        #Note that the first element of the np.where output (a=0) is at the most southern point
        sorted_inds = []
        a = 0

        #Do the coastline walk (NOTE currently a for loop, should be changed to a while loop with a 
        #stopping criteria)
        for _ in range(coastline_masks[l].sum()):

            #Get the x and i indices of the current point
            a_x = bound_xind[a]
            a_y = bound_yind[a]

            #Define a list of all neighbouring points
            neighbours = [[a_x-1,a_y+1],   [a_x,a_y+1],   [a_x+1,a_y+1],
                          [a_x-1,a_y],                    [a_x+1,a_y],
                          [a_x-1,a_y-1],   [a_x,a_y-1],   [a_x+1,a_y-1]]
            
            #Initialise a list of a quantity used to "determine" whether a point is to the left or right of another point:
            # https://stackoverflow.com/a/6989383.
            #The quantity relates to the vector cross-product between the two points relative to the centre.
            #Although the quantity is designed to be either positive or negative depending on if the point is to the 
            # right or left, here we just take the minimum, as in some cases both are positive
            det_ls = []

            #Loop over all neighbouring points
            #NOTE NEED TO FIX THIS FOR CASES THAT INTERSECT WITH THE BOUNDARY. EITHER BY NOT LOOKING FOR THOSE
            #NEIGHBOURING POINTS, OR BY PADDING THE DOMAIN WITH ZEROS, THEN GETTING RID OF THE PAD LATER
            for b in neighbours:
                #If this neighbouring point is a boundary (coastline)
                if coastline_masks[l][b[0],b[1]]:
                    #Check if the point has already been sorted
                    if np.sum( (np.in1d(bound_yind[sorted_inds], b[1])) & (np.in1d(bound_xind[sorted_inds], b[0])) )>=1:
                        det_ls.append(np.nan)
                    #Otherwise, calculate the vector dot product to figure out which neighbout is most to the left of the current point
                    else:
                        det_ls.append((a_x - centre_xind) * (b[1] - centre_yind) - (b[0] - centre_xind) * (a_y - centre_yind))
                else:
                    det_ls.append(np.nan)

            #Choose the neighbouring point to sort based on the minimum cross product function
            #Note that if there are no more neighbouring coastal points that haven't been sorted already,
            #the walk is manually stopped by breaking the for loop
            if np.isnan(det_ls).all():
                break_loop = True
            else:
                break_loop = False
                next_point = neighbours[np.nanargmin(det_ls)]

            if break_loop:
                break

            #Add the point to the sorted list, and update
            a = point_inds[(bound_xind==next_point[0]) & (bound_yind==next_point[1]) & (~np.in1d(point_inds,sorted_inds))][0]
            sorted_inds.append(a)

        sorted_inds_ls.append(sorted_inds)
        bound_xind_ls.append(bound_xind)
        bound_yind_ls.append(bound_yind)

    return [dict(zip(labels,sorted_inds_ls)), 
            dict(zip(labels,bound_xind_ls)), 
            dict(zip(labels,bound_yind_ls))]

def get_coastline_angle_fitted(lsm_ds, r=100, R=300, N=None, small_coast_thresh=0.001):

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

    NOTE The angles output is in degrees from north, ranging from 0 to 180 degrees.
    Because of this, it is not possible to consistently define an onshore/offshore angle using this
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

def rotate_u_v_coast(u, v, angle):

    '''
    Rotate the u and v wind by an angle of a coastline, given in degrees

    Input
    u: U wind component
    v: V wind component
    angle: Angle of orientation of a coastline, in degrees from N

    Output
    uprime: Along-shore wind component
    vprime: Cross-shore wind component

    NOTE that vprime can either be positive or negative for an onshore flow. 
    For eastward-facing coastlines with angles 0-90 degrees from N, positive should be onshore flow
    and for westward-facing coastlines with angles 90-180 degrees from N, positive should be onshore flow
    '''
    
    uprime = (u * np.cos(np.deg2rad(90-angle))) + (v * np.sin(np.deg2rad(90-angle)))
    vprime = (-u * np.sin(np.deg2rad(90-angle))) + (v * np.cos(np.deg2rad(90-angle)))

    return uprime, vprime
    