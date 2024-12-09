import numpy as np
import xarray as xr
import tqdm
import scipy
from skimage.segmentation import find_boundaries, flood, expand_labels
import skimage.measure as measure
import skimage.morphology as morphology
import dask.array as da
from dask.distributed import progress

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

def get_weights(x, p=4, q=4, R=5, slope=-1):
    """
    x the distance
    Let y1 = m1 * (x / R) ** (-p) for x > R.
    Let y2 = S - m2 * (x / R) ** (q) for x <= R.
    Equate y1 and y2 and their derivative at x = R to get
    S = m1 + m2
    slope = -p * m1 = -q * m2 => m1 = -slope/p and m2 = -slope/q
    Thus specifying p, q, R, and the function's slope at x=R determines m1, m2 and S.

    From Ewan Short
    """
    m1 = -slope/p
    m2 = -slope/q
    S = m1 + m2
    y = np.zeros_like(x,dtype=np.float32)
    y[x > R] = m1 * (x[x > R] / R) ** (-p)
    y[x <= R] = S - m2 * (x[x <= R] / R) ** (q)
    y[x==0] = np.nan
    return y

def get_weights_dask(x, p=4, q=4, R=5, slope=-1):
    """

    as in get_weights but using dask
    
    """
    m1 = -slope/p
    m2 = -slope/q
    S = m1 + m2
    #y = da.zeros_like(x,dtype=np.float32)
    y = da.where(x>R,  m1 * (x / R) ** (-p), S - m2 * (x / R) ** (q))
    #y[x > R] = m1 * (x[x > R] / R) ** (-p)
    #y[x <= R] = S - m2 * (x[x <= R] / R) ** (q)
    #y[x==0] = np.nan
    y = da.where(x==0, np.nan, y)
    return y

def fill_coastline_angles(angle_ds):

    '''
    Fill in coastline nan values for the kernel method. Uses nearest neighbour
    Note that angles aren't defined at the coastline for this method
    '''

    # ravel all points and find the valid ones
    points = angle_ds.angle.values.ravel()
    valid = ~np.isnan(points)
    points_valid = points[valid]
    
    # construct arrays of (x, y, z) points, masked to only include the valid points
    xx, yy = np.meshgrid(angle_ds.lon.values, angle_ds.lat.values)
    xx, yy = xx.ravel(), yy.ravel()
    xxv = xx[valid]
    yyv = yy[valid]
    
    # feed these into the interpolator, and also provide the target grid
    interpolated = scipy.interpolate.griddata(np.stack([xxv, yyv]).T, points_valid, (xx, yy), method="nearest")
    
    # reshape to match the original array and replace the DataArray values with
    # the interpolated data
    angle_ds["angle_interp"] = (("lat","lon"),interpolated.reshape(angle_ds.angle.shape))
    angle_ds["coast"] = (("lat","lon"),np.isnan(angle_ds.angle).values * 1)

    return angle_ds    

def get_coastline_angle_kernel(lsm,R=0.2):

    '''
    Ewan's method with help from Jarrah.
    
    Construct a "kernel" for each coastline point based on the anlgle between that point and all other points in the domain, then average..

    Input
    lsm: xarray dataarray with a binary lsm, and lat lon info

    Output
    An xarray dataset with an array of coastline angles (0-360 degrees from N) for the labelled coastline array, as well as an array
    of angle variance as an estimate of how many coastlines are influencing a given point
    '''

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

    #Repeat the 2d lat lon array over a third dimension (corresponding to the coast dim)
    yy_rep=np.repeat(yy[:,:,np.newaxis],yl.shape[0],axis=2)
    xx_rep=np.repeat(xx[:,:,np.newaxis],xl.shape[0],axis=2)

    #Compute the differences in complex space for each coastline points. 
    stack = np.zeros(xx_rep.shape,dtype=np.complex64)
    for t in tqdm.tqdm(range(yl.shape[0])):
        stack[:,:,t] = (yy_rep[:,:,t] - yy_t[t])*1j + (xx_rep[:,:,t] - xx_t[t])    
    del yy_rep,xx_rep
    
    #Reorder to work with the array easier
    stack = np.moveaxis(stack, -1, 0)

    #Get the real part of the complex numbers
    stack_abs = np.abs(stack,dtype=np.float32)
    
    #Create an inverse distance weighting function
    #weights = get_weights(np.abs(stack), p=4, q=2, R=R, slope=-1)
    weights = get_weights(stack_abs, p=4, q=2, R=R, slope=-1)

    #Take the weighted mean and convert complex numbers to an angle and magnitude
    mean_angles = np.mean( (weights*stack) , axis=0)
    mean_abs = np.abs(mean_angles)
    mean_angles = np.angle(mean_angles)    

    #Flip the angles inside the coastline for convention, and convert range to 0 to 2*pi
    mean_angles = np.where(land_label==1,(mean_angles+np.pi) % (2*np.pi),mean_angles % (2*np.pi))

    #Convert angles and magnitude back to complex numbers to do interpolation across the coastline
    mean_complex = mean_abs * np.exp(1j*mean_angles)

    #Do the interpolation across the coastline
    points = mean_complex.ravel()
    valid = ~np.isnan(points)
    points_valid = points[valid]
    xx_rav, yy_rav = xx.ravel(), yy.ravel()
    xxv = xx_rav[valid]
    yyv = yy_rav[valid]
    interpolated_angles = scipy.interpolate.griddata(np.stack([xxv, yyv]).T, points_valid, (xx, yy), method="linear").reshape(lsm.shape)    
    
    #Calculate the weighted circular variance
    total_weight = np.sum(weights, axis=0)
    weights = weights/total_weight
    stack = stack / np.abs(stack)
    variance = 1 - np.abs(np.sum(weights*stack, axis=0))  

    #Reverse the angles for consistency with previous methods, and convert to degrees
    mean_angles = -np.rad2deg(mean_angles) + 360
    interpolated_angles = -np.rad2deg(np.angle(interpolated_angles)) % 360

    #Convert to dataarrays
    angle_da = xr.DataArray(mean_angles,coords={"lat":lat,"lon":lon})
    interpolated_angle_da = xr.DataArray(interpolated_angles,coords={"lat":lat,"lon":lon})
    var_da = xr.DataArray(variance,coords={"lat":lat,"lon":lon})
    coast = xr.DataArray(np.isnan(mean_angles) * 1,coords={"lat":lat,"lon":lon})

    return xr.Dataset({"angle":angle_da,"variance":var_da,"angle_interp":interpolated_angle_da,"coast":coast})

def get_coastline_angle_sorting(lsm,N=10,size_thresh=10,erosion_footprint=morphology.disk(2)):

    '''
    For a land sea mask Dataarray, find the angle of each coastline.
    Here, the resulting coastline is smoothed by erosion (taking the minimum
    pixel over some neighbourhhod, so the coastline is slightly shrinked relative
    to the original land sea mask). Also, the coastline angles are smoothed by taking
    a neighbourhood of N-2 coastal points.

    Input
    lsm: xarray dataarray with a binary lsm, and lat lon info
    N: how many pixels away (both sides) to average the coastline angle (uses (N-1)*2 points)
    size_thresh: labelled land masses with less pixels than this (after erosion) are removed

    Output
    angle: numpy array with coastline points containing angles (0 to 360 degrees from North), and other points as NaNs
    '''

    #Get coastline and labelled landmass masks using skimage
    coastline_masks, label_masks = label_and_find_coast(lsm, 
                                                        erosion_footprint=erosion_footprint,
                                                        size_thresh=size_thresh)
    lon = lsm.lon.values
    lat = lsm.lat.values

    #Sort coastline indices in a counter-clockwise direction
    coastline_masks, label_masks, sorted_inds_ls, bound_xind_ls, bound_yind_ls = order_coastline_points(
        coastline_masks, label_masks)

    #Loop over each labelled land mass in the dictionary, and find angles
    labels = list(label_masks.keys())
    angles = np.zeros(lsm.shape) * np.nan
    next_inds = np.concatenate((np.arange(-N,-1), np.arange(2,N+1)))
    for l in labels:
        bound_xind = bound_xind_ls[l]
        bound_yind = bound_yind_ls[l]
        sorted_inds = sorted_inds_ls[l]
        for ind in range(len(sorted_inds)):
            temp_angles = []
            for n in next_inds:
                if n > 0:
                    temp_angles.append(np.arctan2(
                        lat[bound_xind[sorted_inds[(ind+n) % len(sorted_inds)]]] - 
                            lat[bound_xind[sorted_inds[ind]]],
                        lon[bound_yind[sorted_inds[(ind+n) % len(sorted_inds)]]] - 
                            lon[bound_yind[sorted_inds[ind]]])
                    )
                else:
                    temp_angles.append(np.arctan2(
                        lat[bound_xind[sorted_inds[ind]]] - 
                            lat[bound_xind[sorted_inds[(ind+n)]]],
                        lon[bound_yind[sorted_inds[ind]]] - 
                            lon[bound_yind[sorted_inds[(ind+n)]]]
                            )
                    )

            angles[bound_xind[sorted_inds[ind]], 
                bound_yind[sorted_inds[ind]]] = scipy.stats.circmean(
                    temp_angles,low=-np.pi,high=np.pi)

    angles = np.rad2deg(angles)
    angles = -(angles-90) % 360

    return angles

def expand_coastline_angles(lsm,angles,R=300,N=10,dx=12):

    '''
    From an array of coastline angles, expand the angles out to nearby points, by taking an average of either neighbouring points in a radius, or all points in a radius

    Input
    lsm: a dataarray with a binary land sea mask and lat lon info
    angles: a numpy array of coastline angles ranging from 0-360 degrees (from N)
    R: Radius to expand coastline
    N: number of nearest points to average when expanding the coastal angles (int) by radius R. If None, use all points in radius. If N=1 then the function is very quick, compared
       with if N>1 where it is a nested loop of all points
    dx: the grid spacing of the data in km. Used only if N=1 or N=None
    
    Output
    data_out: xarray dataset with the land sea mask, coastline angles, and expanded coastline angles
    '''

    assert (R is not None) | (N is not None), "R or N must be an integer"

    lon = lsm.lon.values
    lat = lsm.lat.values
    xx,yy = np.meshgrid(lon,lat)

    closest_coast_angle = np.zeros(angles.shape)
    if N is not None:
        print("Expanding angles to average of closest "+str(N)+" points within "+str(R)+" kms...")
        flattened_angles = angles.flatten()
    else:
        print("Expanding angles to average of points within "+str(R)+" kms...")

    if N == 1:
        #Do a nearest neighbour interpolation
        expanded_coast = expand_labels(~np.isnan(angles), distance=np.round(R/dx))
        xa, ya = np.where(~np.isnan(angles))
        points = (xx[xa,ya],yy[xa,ya])
        values = angles[xa,ya]
        xi = (xx.flatten(), yy.flatten())
        closest_coast_angle = np.where(
            expanded_coast,scipy.interpolate.griddata(
                points, values, xi, method="nearest").reshape(xx.shape),
                np.nan)

    elif N is None:
        #If N is set to None, a radius of R km is used to define the coastline
        #angle for each point
        for xi in tqdm.tqdm(range(angles.shape[0])):
            for yi in range(angles.shape[1]):
                expanded_coast = np.zeros(xx.shape)
                expanded_coast[xi,yi] = 1
                expanded_coast = expand_labels(expanded_coast, distance=np.round(R/dx))
                closest_coast_angle[xi,yi] = np.rad2deg(
                    scipy.stats.circmean(
                    np.deg2rad(
                        xr.where((~np.isnan(angles)) & (expanded_coast.astype(bool)),angles,np.nan)
                        ), 
                    high=2*np.pi, low=0, nan_policy="omit"))

    else:
        #If N is not None and not 1, then we are getting the closest N points, and need to calculate
        #acual distances
        # for xi in tqdm.tqdm(range(angles.shape[0])):
        #     for yi in range(angles.shape[1]):
        #         dist = latlon_dist(yy[xi,yi], xx[xi,yi], yy, xx)
        #         if ((~np.isnan(angles)) & (dist<=R)).sum() > 0:
        #             #If N is an integer, then points are assigned a coastline angle using
        #             #the average of the N closest coastline points within R km
        #             closest_coast_angle[xi,yi] = np.rad2deg(
        #                 scipy.stats.circmean(
        #                 np.deg2rad(
        #                     flattened_angles[
        #                         np.argsort(xr.where((~np.isnan(angles)) & (dist<=R),dist,np.nan),axis=None)[0:N]
        #                         ]), 
        #                 high=2*np.pi, low=0, nan_policy="omit"))                        
        #         else:
        #             closest_coast_angle[xi,yi] = np.nan

        #TODO Replace with scipy kdtree

        anglesx, anglesy = np.where(~np.isnan(angles))
        angles_lat = lat[anglesx]
        angles_lon = lon[anglesy]

        X = np.array([angles_lat, angles_lon]).T
        kdt = scipy.spatial.KDTree(X)

        _,ind = kdt.query(np.array([yy.flatten(),xx.flatten()]).T, N)

        closest_coast_angle = np.array(
            [scipy.stats.circmean(
            np.deg2rad(
                angles[ anglesx[ind[i]], anglesy[ind[i]] ])
            ,high=2*np.pi,low=0
            ) for i in tqdm.tqdm(range(ind.shape[0]))]).reshape(xx.shape)

        expanded_coast = expand_labels(~np.isnan(angles), distance=np.round(R/dx))
        closest_coast_angle = np.rad2deg(
            np.where(expanded_coast,closest_coast_angle,np.nan))

    data_out = xr.Dataset({"lsm":lsm,
            "coast_angle_expand":(("lat","lon"),closest_coast_angle),
            "coast_angle":(("lat","lon"),angles)})
    
    return data_out

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
    if erosion_footprint is not None:
        labelled_lsm = measure.label(morphology.binary_opening(lsm,footprint=erosion_footprint),connectivity=1)
    else:
        labelled_lsm = measure.label(lsm,connectivity=1)

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
    For a set of coastline masks, this function uses an algorithm to sort the coastline points in 
    a counter-clockwise direction around the centre of the land mass, starting with the 
    southern-most point for each land mass

    If the coastline has any neighbouring point that is outside of the domain bounds, then the 
    coastline is discarded

    Ouputs are dictionaries of sorted coast indices, as well as the standard (x-y) sorted x inds and y inds,
    and the coastline and label mask dictionaries with land masses removed (that run into the boundary)
    '''
    
    #Get land mass labels from the label masks
    labels = list(label_masks.keys())

    #Keep track of land mass labels to drop if the coastline intersects the domain bounds
    drop_labels = []

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
            
            #If this neighbouring point is a boundary (coastline)
            if (np.max([n[0] for n in neighbours]) >= coastline_masks[l].shape[0])\
                | (np.max([n[1] for n in neighbours]) >= coastline_masks[l].shape[1])\
                    | (np.any([n[0]<0 for n in neighbours]))\
                        | (np.any([n[1]<0 for n in neighbours])):
                neighbours_outside=True
                drop_labels.append(l)
                print("Dropping land mass "+str(l)+" for intersecting boundary")
            else:
                neighbours_outside=False  

            if neighbours_outside:
                break      
            
            #Initialise a list of a quantity used to "determine" whether a point is to the left or right of another point:
            # https://stackoverflow.com/a/6989383.
            #https://en.wikipedia.org/wiki/Cross_product#Computational_geometry
            #The quantity relates to the vector cross-product between the two points relative to the centre.
            #Although the quantity is designed to be either positive or negative depending on if the point is to the 
            # right or left, here we just take the minimum, as in some cases both are positive
            det_ls = []

            #Loop over all neighbouring points
            for b in neighbours:
                if coastline_masks[l][b[0],b[1]]:
                    #Check if the point has already been sorted
                    if np.sum( (np.in1d(bound_yind[sorted_inds], b[1])) & (np.in1d(bound_xind[sorted_inds], b[0])) )>=1:
                        det_ls.append(np.nan)
                    #Otherwise, calculate the vector dot product to figure out which neighbout is most to the left of the current point
                    else:
                        det_ls.append((a_x - centre_xind) * (b[1] - centre_yind) - (b[0] - centre_xind) * (a_y - centre_yind))
                        #det_ls.append(np.sqrt((b[0] - centre_xind)**2 + (b[1] - centre_yind)**2))
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
                #next_point = neighbours[np.nanargmax(det_ls)]

            if break_loop:
                break

            #Add the point to the sorted list, and update
            a = point_inds[(bound_xind==next_point[0]) & (bound_yind==next_point[1]) & (~np.in1d(point_inds,sorted_inds))][0]
            sorted_inds.append(a)

        sorted_inds_ls.append(sorted_inds)
        bound_xind_ls.append(bound_xind)
        bound_yind_ls.append(bound_yind)

    #Store sorted inds (and np.where sorted x and y points) as dictionaries
    sorted_inds_dict = dict(zip(labels,sorted_inds_ls))
    bound_xind_dict = dict(zip(labels,bound_xind_ls))
    bound_yind_dict = dict(zip(labels,bound_yind_ls))

    #Delete all dropped labels from all dicts
    for key in drop_labels:
        del coastline_masks[key]
        del label_masks[key]
        del sorted_inds_dict[key]
        del bound_xind_dict[key]
        del bound_yind_dict[key]

    return coastline_masks, label_masks, sorted_inds_dict, bound_xind_dict, bound_yind_dict

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
    