import xarray as xr
import numpy as np

def interp_np(x, xp, fp):
    return np.interp(x, xp, fp)

def interp_model_level_to_z(z_da,var_da,mdl_dim,heights):

    '''
    Linearly interpolate from model level data to geopotential height levels

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

# interp_temp = xr.apply_ufunc(interp_np,
#                np.array([50,500]),
#                temp["geopotential_hgt_agl"],
#                temp["u_component_of_wind"],
#                input_core_dims=[ ["height"], ["hybrid"], ["hybrid"]],
#                output_core_dims=[["height"]],
#                exclude_dims=set(("hybrid",)),
#                dask="parallelized",
#                output_dtypes=temp["u_component_of_wind"].dtype,
#                vectorize=True)