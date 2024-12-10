from sea_breeze.coastline_funcs import rotate_u_v_coast, latlon_dist
import numpy as np
import xarray as xr
import metpy.calc as mpcalc
import metpy.units as units
import scipy
import tqdm
import dask.array as da

def load_angle_ds(path,lat_slice,lon_slice,chunks="auto"):
    """
    Return an xarray dataset of angles defined by coastline_funcs.get_coastline_angle_kernel()
    e.g. 
    * "/g/data/gb02/ab4502/coastline_data/era5_angles_v2.nc"

    * "/g/data/gb02/ab4502/coastline_data/barra_r_angles_v2.nc"

    * "/g/data/gb02/ab4502/coastline_data/barra_c_angles_v2.nc"

    * "/g/data/gb02/ab4502/coastline_data/era5_global_angles_v2.nc"

    * "/g/data/gb02/ab4502/coastline_data/aus2200_v3.nc"
    """
    return xr.open_dataset(path,chunks=chunks).sel(lat=lat_slice,lon=lon_slice)

def calc_sbi(wind_ds,
                    angle_ds,
                    subtract_mean=False,
                    height_mean=True,
                    blh_da=None,
                    alpha_height=0,
                    sb_heights=[500,2000],
                    mean_heights=[0,4500],
                    height_method="blh",
                    blh_rolling=0,
                    vert_coord="height"):

    '''
    Take a xarray dataset of u and v winds, as well as a dataset of coastline angles, and apply the algorithm of Hallgren et al. (2023) to compute the sea breeze index via a single-column method.

    ### Input
    * wind_ds: An xarray dataset with "u" and "v" wind component variables, and a vertical coordinate "height" in metres

    * angle_ds: An xarray dataset of coastline orientation angles (degrees from N) 

    * subtract_mean: Boolean option for whether to subtract the mean background wind, and calculate perturbation sbi and lbi. Currently using an arithmatic mean over mean_heights layer

    * height_mean: Boolean option to control subtract_mean method. If true, subtract the height mean over mean_heights. Otherwise, subtract the daily mean

    * mean_heights: Array of size (2) that describes the bounds used to mean wind layer. Used if subtract_mean=True

    * blh_da: An xarray dataarray with boundary layer heights, also in m. Used if height_method="blh" to define heights to look for sea/land breezes.

    * alpha_height: Height level in m to define the "low-level" wind

    * sb_heights: Array of size (2) that describes the bounds used to define the upper level sea breeze height. Used if height_method="static"

    * lb_heights: Array of size (2) that describes the bounds used to define the upper level land breeze height. Used if height_method="static"

    * height_method: String used to choose the method for selecting upper level heights to define a circulation. Either "static" or "blh". "static" uses static height limits defined by lb_heights/sb_heights, blh uses the blh_ds.

    * blh_rolling: Integer used to define the number of rolling time windows over which to take the maximum. If zero then no rolling max is taken.

    ### Output
    * xarray dataset with sbi
    '''

    #TODO
    #Check how this looks in different situations (e.g. for a front)

    #Subtract the mean wind. Define mean as the mean over mean_heights m level, or the daily mean
    if subtract_mean:
        print("SUBTRACTING MEAN FROM WINDS FOR SBI CALC...")
        if height_mean:
            u_mean, v_mean = vert_mean_wind(wind_ds,mean_heights,vert_coord)
        else:
            u_mean, v_mean = daily_mean_wind(wind_ds)
        wind_ds["u"] = wind_ds["u"] - u_mean
        wind_ds["v"] = wind_ds["v"] - v_mean

    #Convert coastline orientation angle to the angle perpendicular to the coastline (pointing away from coast. from north)
    theta = (((angle_ds.angle_interp+180)%360-90)%360)

    #Calculate wind directions (from N) for low level (alpha) and all levels (beta)
    def compute_wind_direction(u, v):
        return (90 - np.rad2deg(np.arctan2(-v, -u))) % 360

    alpha = xr.apply_ufunc(
        compute_wind_direction,
        wind_ds["u"].sel({vert_coord: alpha_height}, method="nearest"),
        wind_ds["v"].sel({vert_coord: alpha_height}, method="nearest"),
        dask="parallelized",  # Ensures Dask compatibility
        output_dtypes=[float],  # Specify output dtype
    )

    beta = xr.apply_ufunc(
        compute_wind_direction,
        wind_ds["u"],
        wind_ds["v"],
        dask="parallelized", 
        output_dtypes=[float],  
    )            

    #Calculate the sea breeze index
    def compute_sbi(alpha, beta, theta):
        return (
        np.cos(np.deg2rad(alpha - theta)) *
        np.cos(np.deg2rad(alpha + 180 - beta))
    )
    
    sbi = xr.apply_ufunc(
        compute_sbi,
        alpha, 
        beta,  
        theta,             
        dask="parallelized",  
        output_dtypes=[float],  
    )        

    #Mask to zero everywhere except for the following conditions
    def sbi_conditions(sbi, alpha, beta, theta):
        sb_cond = ( (np.cos(np.deg2rad((alpha - theta)))>0), #Low level flow onshore
            (np.cos(np.deg2rad(beta - (theta+180)))>0), #Upper level flow offshore
            (np.cos(np.deg2rad(alpha + 180 - beta))>0) #Upper level flow opposing
                  )
        return xr.where(sb_cond[0] & sb_cond[1] & sb_cond[2], sbi, 0)
    
    sbi = xr.apply_ufunc(
        sbi_conditions,
        sbi,
        alpha,  # Alpha from the dataset
        beta,   # Beta from the dataset
        theta,             # Theta, could be a scalar or DataArray
        dask="parallelized",  # Enable Dask compatibility
        output_dtypes=[int],  # Specify output dtype
    )        

    #Return the max over some height. Either defined statically or boundary layer height
    _,_,_,hh = da.meshgrid(
        da.rechunk(da.array(wind_ds[wind_ds.u.dims[0]]),chunks={0:wind_ds.u.chunksizes[wind_ds.u.dims[0]][0]}),
        da.rechunk(da.array(wind_ds[wind_ds.u.dims[1]]),chunks={0:wind_ds.u.chunksizes[wind_ds.u.dims[1]][0]}),
        da.rechunk(da.array(wind_ds[wind_ds.u.dims[2]]),chunks={0:wind_ds.u.chunksizes[wind_ds.u.dims[2]][0]}),
        da.rechunk(da.array(wind_ds[wind_ds.u.dims[3]]),chunks={0:wind_ds.u.chunksizes[wind_ds.u.dims[3]][0]}), indexing="ij")
    wind_ds["height_var"] = (wind_ds.u.dims,hh)
    if height_method=="static":
        sbi = xr.where((sbi[vert_coord] >= sb_heights[0]) & (sbi[vert_coord] <= sb_heights[1]),sbi,0)
    elif height_method=="blh":
        if blh_rolling > 0:
            blh_da = blh_da.rolling({"time":blh_rolling}).max()
        sbi = xr.where((wind_ds.height_var <= blh_da),sbi,0)
    else:
        raise ValueError("Invalid height method")

    #TODO: Currently this code below doesn't work well in xarray with dask. I think it is idxmax
    # that forces computation and rechunking. Need to revisit this to make efficient    
    #Calculate the following characteristics of the circulation identified by this method
    #Min height where sbi>0 (bottom of return flow)
    sbi_h_min = xr.where(sbi>0, wind_ds["height_var"], np.nan).min("height")
    #Max height where sbi>0 (top of return flow)
    sbi_h_max = xr.where(sbi>0, wind_ds["height_var"], np.nan).max("height")
    #Height of max sbi (where the return flow most opposes the low level flow)
    sbi_max_inds = sbi.idxmax(dim="height").chunk({"time":sbi.chunksizes["time"][0]})
    sbi_max_h = xr.where(sbi>0, wind_ds["height_var"], np.nan).sel(height=sbi_max_inds)

    # #Same but for the lbi
    # lbi_h_min = xr.where(lbi>0, wind_ds["height_var"], np.nan).min("height")
    # lbi_h_max = xr.where(lbi>0, wind_ds["height_var"], np.nan).max("height")
    # lbi_max_inds = lbi.idxmax(dim="height").chunk({"time":sbi.chunksizes["time"][0]})
    # lbi_max_h = xr.where(lbi>0, wind_ds["height_var"], np.nan).sel(height=lbi_max_inds)

    #Compute each index as the max in the column
    sbi = sbi.max(vert_coord)

    #Dataset output and attributes
    sbi_ds = xr.Dataset({
        "sbi":sbi,
        "sbi_h_min":sbi_h_min,
        "sbi_h_max":sbi_h_max,
        "sbi_max_h":sbi_max_h.drop_vars("height")})
    
    #Set dataset attributes
    sbi_ds = sbi_ds.assign_attrs(
        subtract_mean=str(subtract_mean),
        alpha_height=alpha_height,
        height_method=height_method,
        height_mean=str(height_mean)
    )
    if height_method=="static":
        sbi_ds = sbi_ds.assign_attrs(
            sb_heights=str(sb_heights)
        )    

    #Set dataarray attributes
    sbi_ds["sbi"] = sbi_ds["sbi"].assign_attrs(
        units = "[0,1]",
        long_name = "Sea breeze index",
        description = "This index identifies regions where there is an onshore flow at a near-surface level with an opposing, offshore flow aloft in the boundary layer. The SBI is calculated for each vertical layer and then the maximum is taken. Following Hallgren et al. 2023 (10.1175/WAF-D-22-0163.1).")  
    sbi_ds["sbi_h_min"] = sbi_ds["sbi_h_min"].assign_attrs(
        units = "m",
        long_name = "Minimum height where sbi is positive"
    )
    sbi_ds["sbi_h_max"] = sbi_ds["sbi_h_max"].assign_attrs(
            units = "m",
            long_name = "Maximum height where sbi is positive"
        )    
    sbi_ds["sbi_max_h"] = sbi_ds["sbi_max_h"].assign_attrs(
                units = "m",
                long_name = "Height of maximum sbi"
            )        
    
    return sbi_ds

def calc_lbi(wind_ds,
                    angle_ds,
                    subtract_mean=False,
                    height_mean=True,
                    blh_da=None,
                    alpha_height=0,
                    lb_heights=[100,900],
                    mean_heights=[0,4500],
                    height_method="blh",
                    blh_rolling=0,
                    vert_coord="height"):

    '''
    Take a xarray dataset of u and v winds, as well as a dataset of coastline angles, and apply the algorithm of Hallgren et al. (2023) to compute the land breeze index via a single-column method.

    ### Input
    * wind_ds: An xarray dataset with "u" and "v" wind component variables, and a vertical coordinate "height" in metres

    * angle_ds: An xarray dataset of coastline orientation angles (degrees from N) 

    * subtract_mean: Boolean option for whether to subtract the mean background wind, and calculate perturbation sbi and lbi. Currently using an arithmatic mean over mean_heights layer

    * height_mean: Boolean option to control subtract_mean method. If true, subtract the height mean over mean_heights. Otherwise, subtract the daily mean

    * mean_heights: Array of size (2) that describes the bounds used to mean wind layer. Used if subtract_mean=True

    * blh_da: An xarray dataarray with boundary layer heights, also in m. Used if height_method="blh" to define heights to look for sea/land breezes.

    * alpha_height: Height level in m to define the "low-level" wind

    * sb_heights: Array of size (2) that describes the bounds used to define the upper level sea breeze height. Used if height_method="static"

    * lb_heights: Array of size (2) that describes the bounds used to define the upper level land breeze height. Used if height_method="static"

    * height_method: String used to choose the method for selecting upper level heights to define a circulation. Either "static" or "blh". "static" uses static height limits defined by lb_heights/sb_heights, blh uses the blh_ds.

    * blh_rolling: Integer used to define the number of rolling time windows over which to take the maximum. If zero then no rolling max is taken.

    ### Output
    * xarray dataset with sbi and lbi
    '''

    #TODO
    #Check how this looks in different situations (e.g. for a front)

    #Subtract the mean wind. Define mean as the mean over mean_heights m level, or the daily mean
    if subtract_mean:
        if height_mean:
            u_mean, v_mean = vert_mean_wind(wind_ds,mean_heights,vert_coord)
        else:
            u_mean, v_mean = daily_mean_wind(wind_ds)
        wind_ds["u"] = wind_ds["u"] - u_mean
        wind_ds["v"] = wind_ds["v"] - v_mean

    #Convert coastline orientation angle to the angle perpendicular to the coastline (pointing away from coast. from north)
    theta = (((angle_ds.angle_interp+180)%360-90)%360)

    #Calculate wind directions (from N) for low level (alpha) and all levels (beta)
    alpha = (90 - np.rad2deg(np.arctan2(
        -wind_ds["v"].sel(height=alpha_height,method="nearest"),
        -wind_ds["u"].sel(height=alpha_height,method="nearest")))) % 360
    beta = (90 - np.rad2deg(np.arctan2(
        -wind_ds["v"], 
        -wind_ds["u"]))) % 360
    wind_ds["alpha"] = alpha
    wind_ds["beta"] = beta

    #Calculate the sea breeze and land breeze indices
    lbi = -np.cos(np.deg2rad((wind_ds.alpha - theta))) * \
            np.cos(np.deg2rad(wind_ds.alpha + 180 - wind_ds.beta))

    #Mask to zero everywhere except for the following conditions
    lb_cond = ( (np.cos(np.deg2rad((wind_ds.alpha - (theta+180))))>0), #Low level flow offshore
        (np.cos(np.deg2rad(wind_ds.beta - theta))>0), #Upper level flow onshore
        (np.cos(np.deg2rad(wind_ds.alpha + 180 - wind_ds.beta))>0) #Upper level flow opposing
              )
    lbi = xr.where(lb_cond[0] & lb_cond[1] & lb_cond[2], lbi, 0)

    #Return the max over some height. Either defined statically or boundary layer height
    #_,_,_,hh = np.meshgrid(wind_ds.time,wind_ds.lat,wind_ds.lon,wind_ds.height)
    _,_,_,hh = da.meshgrid(
        da.rechunk(da.array(wind_ds[wind_ds.u.dims[0]]),chunks={0:wind_ds.u.chunksizes[wind_ds.u.dims[0]][0]}),
        da.rechunk(da.array(wind_ds[wind_ds.u.dims[1]]),chunks={0:wind_ds.u.chunksizes[wind_ds.u.dims[1]][0]}),
        da.rechunk(da.array(wind_ds[wind_ds.u.dims[2]]),chunks={0:wind_ds.u.chunksizes[wind_ds.u.dims[2]][0]}),
        da.rechunk(da.array(wind_ds[wind_ds.u.dims[3]]),chunks={0:wind_ds.u.chunksizes[wind_ds.u.dims[3]][0]}), indexing="ij")
    wind_ds["height_var"] = (wind_ds.u.dims,hh)
    if height_method=="static":
        lbi = xr.where((lbi.height >= lb_heights[0]) & (lbi.height <= lb_heights[1]),lbi,0)
    elif height_method=="blh":
        if blh_rolling > 0:
            blh_da = blh_da.rolling({"time":blh_rolling}).max()
        lbi = xr.where((wind_ds.height_var <= blh_da),lbi,0)
    else:
        raise ValueError("Invalid height method")

    #TODO: Currently this code below doesn't work well in xarray with dask. I think it is idxmax
    # that forces computation and rechunking. Need to revisit this to make efficient    
    #Calculate the following characteristics of the circulation identified by this method
    # #Min height where sbi>0 (bottom of return flow)
    # sbi_h_min = xr.where(sbi>0, wind_ds["height_var"], np.nan).min("height")
    # #Max height where sbi>0 (top of return flow)
    # sbi_h_max = xr.where(sbi>0, wind_ds["height_var"], np.nan).max("height")
    # #Height of max sbi (where the return flow most opposes the low level flow)
    # sbi_max_inds = sbi.idxmax(dim="height").chunk({"time":sbi.chunksizes["time"][0]})
    # sbi_max_h = xr.where(sbi>0, wind_ds["height_var"], np.nan).sel(height=sbi_max_inds)

    # #Same but for the lbi
    # lbi_h_min = xr.where(lbi>0, wind_ds["height_var"], np.nan).min("height")
    # lbi_h_max = xr.where(lbi>0, wind_ds["height_var"], np.nan).max("height")
    # lbi_max_inds = lbi.idxmax(dim="height").chunk({"time":sbi.chunksizes["time"][0]})
    # lbi_max_h = xr.where(lbi>0, wind_ds["height_var"], np.nan).sel(height=lbi_max_inds)

    #Compute each index as the max in the column
    lbi = lbi.max("height")

    #Dataset output
    # sbi_ds = xr.Dataset({
    #     "sbi":sbi,
    #     "sbi_h_min":sbi_h_min,
    #     "sbi_h_max":sbi_h_max,
    #     "sbi_max_h":sbi_max_h.drop_vars("height"),
    #     "lbi":lbi,
    #     "lbi_h_max":lbi_h_max,
    #     "lbi_h_min":lbi_h_min,
    #     "lbi_max_h":lbi_max_h.drop_vars("height")
    # }
    # )
    sbi_ds = xr.Dataset({"lbi":lbi})        

    return sbi_ds

def moisture_flux_gradient(q, u, v, angle_ds, lat_chunk="auto", lon_chunk="auto"):

    """
    Calculate d(qu)/dt

    ## Input
    * q: xarray dataarray of specific humidity in kg/kg

    * u: xarray dataarray of u winds in m/s

    * v: xarray dataarray of v winds in m/s

    * angle_ds: xarray dataset containing coastline angles ("angle_interp")

    ## Output:
    * xarray dataset
    """

    #Rechunk data in one time dim
    q = q.chunk({"time":-1,"lat":lat_chunk,"lon":lon_chunk})
    u = u.chunk({"time":-1,"lat":lat_chunk,"lon":lon_chunk})
    v = v.chunk({"time":-1,"lat":lat_chunk,"lon":lon_chunk})

    #Convert hus to g/kg 
    q = q * 1000
    
    #Define angle of coastline orientation from N
    theta=angle_ds.angle_interp 
    
    #Rotate angle to be perpendicular to theta, from E (i.e. mathamatical angle definition)
    rotated_angle=(((theta)%360-90)%360) + 90   
    
    #Define normal angle vectors, pointing onshore
    cx, cy = [-np.cos(np.deg2rad(rotated_angle)), np.sin(np.deg2rad(rotated_angle))]
    
    #Calculate the wind component perpendicular and parallel to the coast by using the normal unit vectors
    vprime = ((u*cx) + (v*cy))

    #Calculate the rate of change
    dqu_dt = (vprime*q).differentiate("time",datetime_unit="s")

    return dqu_dt    

def kinematic_frontogenesis(q,u,v):

    """
    Calculate 2d kinematic frontogenesis, with water vapour mixing ratio
    Will identify all regions where moisture fronts are increasing/decreasing due to deformation/convergence, including potentially sea breeze fronts

    Uses metpy formulation but in numpy/xarray, for efficiency
    https://github.com/Unidata/MetPy/blob/756ce975eb25d17827924da21d5d77f38a184bd4/src/metpy/calc/kinematics.py#L478

    Inputs
    * q: xarray dataarray of water vapour mixing ratio (although this function should work with any scalar). Expects lat/lon/time coordinates in units kg/kg

    * u: as above for a u wind component

    * v: as above for a v wind component

    * subtract_mean: boolean option for whether to subtract the mean background wind, and calculate frontogenesis using perturbation winds. mean defined by a vertical avg over mean_heights

    * weighted_mean: if true, take the vertical mean weighted by the vertical coordinate 

    * wind_ds: dataset with u and v winds on vertical levels

    * mean_heights: heights between which to average, in unite of vert_coord

    * vert_coord: if wind_ds is being used to take a vertical average, what is the vertical coord

    Returns
    * 2d kinematic frontogenesis in units (g/kg) / 100 km / 3h    """

    #Convert specific humidity to g/kg
    q = q*1000

    #Calculate grid spacing in km using metpy, in x and y
    x, y = np.meshgrid(q.lon,q.lat)
    dx, dy = mpcalc.lat_lon_grid_deltas(x,y)

    #Convert the x and y grid spacing arrays into xarray datasets. Need to interpolate to match the original grid
    dx = xr.DataArray(np.array(dx),dims=["lat","lon"],coords={"lat":q.lat.values, "lon":q.lon.values[0:-1]}).\
            interp({"lon":q.lon,"lat":q.lat},method="linear",kwargs={"fill_value":"extrapolate"}).\
            chunk({"lat":q.chunksizes["lat"][0], "lon":q.chunksizes["lon"][0]})
    dy = xr.DataArray(np.array(dy),dims=["lat","lon"],coords={"lat":q.lat.values[0:-1], "lon":q.lon.values}).\
            interp({"lon":q.lon,"lat":q.lat},method="linear",kwargs={"fill_value":"extrapolate"}).\
            chunk({"lat":q.chunksizes["lat"][0], "lon":q.chunksizes["lon"][0]})

    #Calculate horizontal moisture gradient
    ddy_q = (xr.DataArray(da.gradient(q,axis=q.get_axis_num("lat")), dims=q.dims, coords=q.coords) / dy)
    ddx_q = (xr.DataArray(da.gradient(q,axis=q.get_axis_num("lon")), dims=q.dims, coords=q.coords) / dx)
    mag_dq = np.sqrt( ddy_q**2 + ddx_q**2)

    #Calculate horizontal U and V gradients, as well as divergence and deformation 
    #Following https://www.ncl.ucar.edu/Document/Functions/Contributed/shear_stretch_deform.shtml
    ddy_u = (xr.DataArray(da.gradient(u,axis=q.get_axis_num("lat")), dims=q.dims, coords=q.coords) / dy)
    ddx_u = (xr.DataArray(da.gradient(u,axis=q.get_axis_num("lon")), dims=q.dims, coords=q.coords) / dx)
    ddy_v = (xr.DataArray(da.gradient(v,axis=q.get_axis_num("lat")), dims=q.dims, coords=q.coords) / dy)
    ddx_v = (xr.DataArray(da.gradient(v,axis=q.get_axis_num("lon")), dims=q.dims, coords=q.coords) / dx)
    div = ddx_u + ddy_v
    strch_def = ddx_u - ddy_v
    shear_def = ddx_v + ddy_u
    tot_def = np.sqrt(strch_def**2 + shear_def**2)

    #Calculate the angle between axis of dilitation and isentropes
    psi = 0.5 * np.arctan2(shear_def, strch_def)
    beta = np.arcsin((-ddx_q * np.cos(psi) - ddy_q * np.sin(psi)) / mag_dq)

    return 0.5 * mag_dq * (tot_def * np.cos(2 * beta) - div) * 1.08e9

def kinematic_frontogenesis_metpy(q,u,v,subtract_mean=False,weighted_mean=True,wind_ds=None,mean_heights=[0,4500],vert_coord="level"):

    '''
    Use the frontogenesis metpy function to calculate 2d kinematic frontogenesis, with water vapour mixing ratio
    Will identify all regions where moisture fronts are increasing/decreasing due to deformation/convergence, including potentially sea breeze fronts

    Inputs
    * q: xarray dataarray of water vapour mixing ratio (although this function should work with any scalar). Expects lat/lon/time coordinates in units kg/kg

    * u: as above for a u wind component

    * v: as above for a v wind component

    * subtract_mean: boolean option for whether to subtract the mean background wind, and calculate frontogenesis using perturbation winds. mean defined by a vertical avg over mean_heights

    * weighted_mean: if true, take the vertical mean weighted by the vertical coordinate 

    * wind_ds: dataset with u and v winds on vertical levels

    * mean_heights: heights between which to average, in unite of vert_coord

    * vert_coord: if wind_ds is being used to take a vertical average, what is the vertical coord

    Returns
    * 2d kinematic frontogenesis in units (kg/kg) / 100 km / 3h
    '''

    if hasattr(q, "units"):
        if q.units == "K":
            pass
        else:
            q = q * units.units("K")

    #Convert to g/kg
    q = q * 1000 

    if hasattr(u, "units"):
        if (u.units == "m/s") | (u.units ==  'm s-1'):
            pass
        else:
            u = u * units.units("m/s")

    if hasattr(v, "units"):
        if (v.units == "m/s") | (v.units ==  'm s-1'):
            pass
        else:
            v = v * units.units("m/s")                        

    if subtract_mean:
        if weighted_mean:
            u_mean, v_mean = weighted_vert_mean_wind(wind_ds,mean_heights,wind_ds[vert_coord],vert_coord)
        else:
            u_mean, v_mean = vert_mean_wind(wind_ds,mean_heights,vert_coord)
        u = u - u_mean
        v = v - v_mean

    Fq = mpcalc.frontogenesis(q,
                            u,
                            v,
                            x_dim=q.get_axis_num("lon"),
                            y_dim=q.get_axis_num("lat")) * 1.08e9
    
    Fq["units"] = "g/kg/100km/3hr"
    
    return Fq

def coast_relative_frontogenesis(q,u,v,angle_ds):

    """
    This function calculates 2d shearing and confluence in the moisture field, using a wind field that has been rotated to be coastline-relative.

    ## Input:
    * q: xarray dataarray of specific humidity in kg/kg

    * u: xarray dataarray of u winds in m/s

    * v: xarray dataarray of v winds in m/s

    * angle_ds: xarray dataset containing coastline angles ("angle_interp")

    ## Output:
    * xarray dataset with variables of "shearing" and "confluence". These describe the changes in the moisture gradient due to strecthing and shearing relative to the coastline.
    
    """

    #Convert to g/kg
    q = q * 1000 

    #Calculate grid spacing in km using metpy, in x and y
    x, y = np.meshgrid(q.lon,q.lat)
    dx, dy = mpcalc.lat_lon_grid_deltas(x,y)

    #Convert the x and y grid spacing arrays into xarray datasets. Need to interpolate to match the original grid
    dx = xr.DataArray(np.array(dx),dims=["lat","lon"],coords={"lat":q.lat.values, "lon":q.lon.values[0:-1]}).\
            interp({"lon":q.lon,"lat":q.lat},method="linear",kwargs={"fill_value":"extrapolate"}).\
            chunk({"lat":q.chunksizes["lat"][0], "lon":q.chunksizes["lon"][0]})
    dy = xr.DataArray(np.array(dy),dims=["lat","lon"],coords={"lat":q.lat.values[0:-1], "lon":q.lon.values}).\
            interp({"lon":q.lon,"lat":q.lat},method="linear",kwargs={"fill_value":"extrapolate"}).\
            chunk({"lat":q.chunksizes["lat"][0], "lon":q.chunksizes["lon"][0]})

    #Define angle of coastline orientation from N
    theta=angle_ds.angle_interp 

    #Rotate angle to be perpendicular to theta, from E (i.e. mathamatical angle definition)
    rotated_angle=(((theta)%360-90)%360) + 90   

    #Define normal angle vectors, pointing onshore
    cx, cy = [-np.cos(np.deg2rad(rotated_angle)), np.sin(np.deg2rad(rotated_angle))]

    #Define normal angle vectors, pointing alongshore
    ax, ay = [-np.cos(np.deg2rad(rotated_angle - 90)), np.sin(np.deg2rad(rotated_angle - 90))]    

    #Calculate the wind component perpendicular and parallel to the coast by using the normal unit vectors
    vprime = ((u*cx) + (v*cy))
    uprime = ((u*ax) + (v*ay))

    #Calculate the gradients of moisture, and (rotated) winds in x/y coordinates
    # dq_dx, dq_dy = mpcalc.geospatial_gradient((q * units.units("g/g")).metpy.convert_units("g/kg"), x_dim=q.get_axis_num("lon"), y_dim=q.get_axis_num("lat"))
    # dvprime_dx, dvprime_dy = mpcalc.geospatial_gradient(vprime*units.units("m/s"), x_dim=vprime.get_axis_num("lon"), y_dim=vprime.get_axis_num("lat"))
    # duprime_dx, duprime_dy = mpcalc.geospatial_gradient(uprime*units.units("m/s"), x_dim=uprime.get_axis_num("lon"), y_dim=uprime.get_axis_num("lat"))

    #Calculate horizontal moisture gradients
    dq_dx = (xr.DataArray(da.gradient(q,axis=q.get_axis_num("lon")), dims=q.dims, coords=q.coords) / dx)
    dq_dy = (xr.DataArray(da.gradient(q,axis=q.get_axis_num("lat")), dims=q.dims, coords=q.coords) / dy)        

    #Calculate onshore and alongshore wind gradients
    dvprime_dx = (xr.DataArray(da.gradient(vprime,axis=q.get_axis_num("lon")), dims=q.dims, coords=q.coords) / dx)
    dvprime_dy = (xr.DataArray(da.gradient(vprime,axis=q.get_axis_num("lat")), dims=q.dims, coords=q.coords) / dy)      
    duprime_dx = (xr.DataArray(da.gradient(uprime,axis=q.get_axis_num("lon")), dims=q.dims, coords=q.coords) / dx)
    duprime_dy = (xr.DataArray(da.gradient(uprime,axis=q.get_axis_num("lat")), dims=q.dims, coords=q.coords) / dy)            

    #Rotate gradients to cross shore (c) and along shore (a)
    dq_dc = (dq_dx*cx.values) + (dq_dy*cy.values)    
    dvprime_dc = (dvprime_dx*cx.values) + (dvprime_dy*cy.values)
    duprime_dc = (duprime_dx*cx.values) + (duprime_dy*cy.values)
    dq_da = (dq_dx*ax.values) + (dq_dy*ay.values)    

    #Calculate the gradient in moisture convergence, convert to a Dataarray, and return
    confluence = dq_dc * dvprime_dc
    shearing = dq_da * duprime_dc
    
    #Format for output and calculate the shearing plus confluence
    out =  xr.Dataset({
        "confluence":xr.DataArray(confluence,coords=q.coords),
        "shearing":xr.DataArray(shearing,coords=q.coords)})
    out["total"] = out["shearing"] + out["confluence"]

    return out

def weighted_vert_mean_wind(wind_ds,mean_heights,p,vert_coord):

    """
    For an xarray dataset with u and v winds, take the vertical weighted mean over some layer.
    Weight by pressure (p)
    """

    p_slice = p.sel({vert_coord:slice(mean_heights[0],mean_heights[1])})

    u_mean = (wind_ds["u"] * p).sel({vert_coord:slice(mean_heights[0],mean_heights[1])}).sum(vert_coord) / np.sum(p_slice)
    v_mean = (wind_ds["v"] * p).sel({vert_coord:slice(mean_heights[0],mean_heights[1])}).sum(vert_coord) / np.sum(p_slice)

    return u_mean.persist(), v_mean.persist()

def vert_mean_wind(wind_ds,mean_heights,vert_coord):

    """
    For an xarray dataset with u and v winds, take the vertical mean over some layer.
    """

    u_mean = wind_ds["u"].sel({vert_coord:slice(mean_heights[0],mean_heights[1])}).mean(vert_coord)
    v_mean = wind_ds["v"].sel({vert_coord:slice(mean_heights[0],mean_heights[1])}).mean(vert_coord)

    return u_mean, v_mean

def daily_mean_wind(wind_ds):

    """
    For an xarray dataset with u and v winds, take a rolling daily mean
    """

    dt_h = np.round((wind_ds.time.diff("time")[0].values / (1e9 * 60 * 60)).astype(float)).astype(int)
    time_window = int(24 / dt_h)
    min_periods = int(time_window/2)

    u_mean = wind_ds["u"].rolling(dim={"time":time_window},center=True,min_periods=min_periods).mean()
    v_mean = wind_ds["v"].rolling(dim={"time":time_window},center=True,min_periods=min_periods).mean()

    return u_mean, v_mean

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

        raise ValueError("weighting must equal ''none'' or ''gaussian''")

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