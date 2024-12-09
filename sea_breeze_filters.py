import skimage
import scipy
import tqdm
import numpy as np

def binary_mask(field,thresh):

    """
    Take a field that diagnoses sea breeze potential and create a binary sea breeze mask based on a threshold

    Absolute versus percentile/perturbation thresholds? Standard deviation threshold
    """


def filter_sea_breeze(mask,options,angle_ds,ta):
    
    """
    Take a binary sea breeze mask and identify objects, then filter it for sea breezes based on several conditions related to those objects

    - Area (number of pixels/spatial area).
    - Eccentricity or aspect ratio.
    - Orientation of object relative to the coastline.
    - Within some distance to the coast.
    - Positive land-sea temperature contrast/gradient.

    - Propagation speed? Note this is not straightforward as this assumes several time steps are available. Somewhat limits applications because
    we need at least one day of data at a time.

    """

    labels = skimage.measure.label(mask)
    region_props = skimage.measure.regionprops(labels,spacing=(1,1))

    labs = np.array([region_props[i].label for i in np.arange(len(region_props))])
    eccen = [region_props[i].eccentricity for i in np.arange(len(region_props))]
    area = [region_props[i].area for i in np.arange(len(region_props))]
    orient = np.rad2deg(np.array([region_props[i].orientation for i in np.arange(len(region_props))]))

    angle_tol=20
    eccen_thresh=0.95
    area_thresh=10

    to_remove = labs[(np.array(eccen) < eccen_thresh)].squeeze()
    to_remove = list(np.concatenate([to_remove,labs[(np.array(area) < area_thresh)].squeeze()]))    
    to_remove = np.unique(to_remove)

    for i in to_remove:
        labels[labels==i] = 0

    #Remove objects from labs array and orient array
    #TODO speed this up using groupby and apply? xr.apply_ufunc(scipyfunc,groupby_obj,parallel,output_dtype)
    coast_angles = [scipy.stats.circmean(angle_ds.angle_interp[labels==l],low=-90,high=90) for l in tqdm.tqdm(labs)]
    orient_diff = np.abs(orient-coast_angles) < angle_tol

    #Remove objects based on orient_diff