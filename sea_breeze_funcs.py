import numpy as np
import xarray as xr
import metpy.calc as mpcalc
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
                    angle_da,
                    subtract_mean=False,
                    height_mean=False,
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

    * angle_da: An xarray dataarray of coastline orientation angles (degrees from N) 

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
    theta = (((angle_da+180)%360-90)%360)

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

    #Mask to zero everywhere except for the following conditions. Keep masked if angles are nans
    def sbi_conditions(sbi, alpha, beta, theta):
        sb_cond = ( (np.cos(np.deg2rad((alpha - theta)))>0), #Low level flow onshore
            (np.cos(np.deg2rad(beta - (theta+180)))>0), #Upper level flow offshore
            (np.cos(np.deg2rad(alpha + 180 - beta))>0) #Upper level flow opposing
                  )
        sbi_cond = xr.where(sb_cond[0] & sb_cond[1] & sb_cond[2], sbi, 0)
        return xr.where(np.isnan(theta),np.nan,sbi_cond)
    
    sbi = xr.apply_ufunc(
        sbi_conditions,
        sbi,
        alpha,  
        beta,   
        theta,             
        dask="parallelized",  
        output_dtypes=[float],  
    )        

    #Return the max over some height, keeping masked if angles are nans. Height layer either defined statically or using boundary layer height
    time_dim = wind_ds.u.get_axis_num("time")
    lat_dim = wind_ds.u.get_axis_num("lat")
    lon_dim = wind_ds.u.get_axis_num("lon")
    height_dim = wind_ds.u.get_axis_num(vert_coord)
    _,hh,_,_ = da.meshgrid(
        da.rechunk(da.array(wind_ds[wind_ds.u.dims[time_dim]]),chunks={0:wind_ds.u.chunksizes[wind_ds.u.dims[time_dim]][0]}),
        da.rechunk(da.array(wind_ds[wind_ds.u.dims[height_dim]]),chunks={0:wind_ds.u.chunksizes[wind_ds.u.dims[height_dim]][0]}),
        da.rechunk(da.array(wind_ds[wind_ds.u.dims[lat_dim]]),chunks={0:wind_ds.u.chunksizes[wind_ds.u.dims[lat_dim]][0]}),
        da.rechunk(da.array(wind_ds[wind_ds.u.dims[lon_dim]]),chunks={0:wind_ds.u.chunksizes[wind_ds.u.dims[lon_dim]][0]}), indexing="ij")
    wind_ds["height_var"] = (("time",vert_coord,"lat","lon"),hh)
    if height_method=="static":
        sbi = xr.where((sbi[vert_coord] >= sb_heights[0]) & (sbi[vert_coord] <= sb_heights[1]),sbi,0)
    elif height_method=="blh":
        if blh_rolling > 0:
            blh_da = blh_da.rolling({"time":blh_rolling}).max()
        sbi = xr.where((wind_ds.height_var <= blh_da),sbi,0)
    else:
        raise ValueError("Invalid height method")
    sbi = xr.where(np.isnan(theta),np.nan,sbi)

    #Height of max sbi (where the return flow most opposes the low level flow)
    #sbi_max_h = sbi.idxmax(dim=vert_coord)

    #Compute each index as the max in the column
    sbi = sbi.max(vert_coord)

    #Dataset output and attributes
    sbi_ds = xr.Dataset({
        "sbi":sbi,})#.persist()
        #"sbi_max_h":sbi_max_h})
    
    #Set dataset attributes
    sbi_ds = sbi_ds.assign_attrs(
        subtract_mean=str(subtract_mean),
        alpha_height=alpha_height,
        height_method=height_method,
        height_mean=str(height_mean),
        hgt_levs=str(wind_ds[vert_coord].values),
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

    # sbi_ds["sbi_max_h"] = sbi_ds["sbi_max_h"].assign_attrs(
    #             units = "m",
    #             long_name = "Height of maximum sbi"
    #         )        
    
    return sbi_ds

def moisture_flux_gradient(q, u, v, angle_da, lat_chunk="auto", lon_chunk="auto"):

    """
    DEPRECATED: Use hourly_change instead

    Calculate d(qu)/dt

    ## Input
    * q: xarray dataarray of specific humidity in kg/kg

    * u: xarray dataarray of u winds in m/s

    * v: xarray dataarray of v winds in m/s

    * angle_da: xarray dataset containing coastline angles ("angle_interp")

    ## Output:
    * xarray dataset
    """

    #Rechunk data in one time dim
    # q = q.chunk({"time":-1,"lat":lat_chunk,"lon":lon_chunk})
    # u = u.chunk({"time":-1,"lat":lat_chunk,"lon":lon_chunk})
    # v = v.chunk({"time":-1,"lat":lat_chunk,"lon":lon_chunk})

    #Convert hus to g/kg 
    q = q * 1000
    
    #Define angle of coastline orientation from N
    theta=angle_da 
    
    #Rotate angle to be perpendicular to theta, from E (i.e. mathamatical angle definition)
    rotated_angle=(((theta)%360-90)%360) + 90   
    
    #Define normal angle vectors, pointing onshore
    cx, cy = [-np.cos(np.deg2rad(rotated_angle)), np.sin(np.deg2rad(rotated_angle))]
    
    #Calculate the wind component perpendicular and parallel to the coast by using the normal unit vectors
    vprime = ((u*cx) + (v*cy))

    #Calculate the rate of change
    dqu_dt = (vprime*q).differentiate("time",datetime_unit="s")

    #Convert to xr dataset and assign attributes
    ds = xr.Dataset({"dqu_dt":dqu_dt}).persist()
    ds["dqu_dt"] = ds["dqu_dt"].assign_attrs(
        units = "g/kg/m/(s^2)",
        long_name = "Moisture flux rate of change",
        description = "Rate of change of onshore moisture flux gradient (dqu/dt).")      

    return ds

def rotate_wind(u,v,theta):

    """
    Rotate u and v wind components to be cross-shore and along-shore, based on angle of coastline orientation, theta.

    ## Input
    * u: xarray dataarray of u winds in m/s
    * v: xarray dataarray of v winds in m/s
    * theta: xarray dataarray of coastline orientation angles from N

    ## Output
    * uprime: xarray dataarray of wind component parallel to the coast
    * vprime: xarray dataarray of wind component perpendicular to the coast
    """

    #Rotate angle to be perpendicular to theta, from E (i.e. mathamatical angle definition)
    rotated_angle=(((theta)%360-90)%360) + 90   
    
    #Define normal angle vectors, pointing onshore
    cx, cy = [-np.cos(np.deg2rad(rotated_angle)), np.sin(np.deg2rad(rotated_angle))]
    
    #Define normal angle vectors, pointing alongshore
    ax, ay = [-np.cos(np.deg2rad(rotated_angle - 90)), np.sin(np.deg2rad(rotated_angle - 90))]    
    
    #Calculate the wind component perpendicular and parallel to the coast by using the normal unit vectors
    uprime = ((u*ax) + (v*ay))
    vprime = ((u*cx) + (v*cy))

    return uprime, vprime    

def hourly_change(q, t, u, v, angle_da, lat_chunk="auto", lon_chunk="auto"):

    """
    Calculate hourly changes in q, t, and onshore wind speed. Use thresholds on each to define candidate sea breezes 

    ## Input
    * q: xarray dataarray of specific humidity in kg/kg

    * t: xarray dataarray of air temperature in degrees

    * u: xarray dataarray of u winds in m/s

    * v: xarray dataarray of v winds in m/s

    * angle_da: xarray dataset containing coastline angles ("angle_interp")

    ## Output:
    * xarray dataset
    """

    #Rechunk data in one time dim
    q = q.chunk({"time":-1,"lat":lat_chunk,"lon":lon_chunk})
    u = u.chunk({"time":-1,"lat":lat_chunk,"lon":lon_chunk})
    v = v.chunk({"time":-1,"lat":lat_chunk,"lon":lon_chunk})
    t = t.chunk({"time":-1,"lat":lat_chunk,"lon":lon_chunk})

    #Convert hus to g/kg 
    q = q * 1000
    
    #Rotate the winds to be cross-shore (uprime) and along-shore (vprime)
    uprime, vprime = rotate_wind(u,v,angle_da)

    #Calculate the rate of change
    #dqu_dt = (vprime*q).differentiate("time",datetime_unit="s")
    wind_change = vprime.diff("time")
    q_change = q.diff("time")
    t_change = t.diff("time")  

    if "height" in list(wind_change.coords.keys()):
        wind_change = wind_change.drop_vars("height")
    if "height" in list(wind_change.coords.keys()):        
        t_change = t_change.drop_vars("height")
    if "height" in list(q_change.coords.keys()):        
        q_change = q_change.drop_vars("height")     

#    ds = xr.Dataset(
#        {"wind_change":wind_change,
#         "q_change":q_change,
#         "t_change":t_change,
#         "dqu_dt":dqu_dt
#            }).persist()
    ds = xr.Dataset(
        {"wind_change":wind_change,
         "q_change":q_change,
         "t_change":t_change,
            }).persist()
    ds["wind_change"] = ds["wind_change"].assign_attrs(
        units = "m/s/h",
        long_name = "Onshore wind speed rate of change",
        description = "Rate of change of onshore wind speed in m/s/h.")
    ds["t_change"] = ds["t_change"].assign_attrs(
        units = "K/h",
        long_name = "Local temperature rate of change.",
        description = "Rate of change of temperature.")      
    ds["q_change"] = ds["q_change"].assign_attrs(
        units = "g/kg/h",
        long_name = "Local specific humidity rate of change.",
        description = "Rate of change of specific humidity.")
    # ds["dqu_dt"] = ds["dqu_dt"].assign_attrs(
    #     units = "g/kg/m/(s^2)",
    #     long_name = "Moisture flux rate of change",
    #     description = "Rate of change of onshore moisture flux gradient (dqu/dt).")    
    
    return ds

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

    # x_stacked = da.array(x).flatten()
    # x2 = da.percentile(x_stacked,100,internal_method="tdigest") / D

    x2 = (x.max() / D).persist()
    #x2 = 1
    
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

    Returns
    * 2d kinematic frontogenesis in units (g/kg) / 100 km / 3h    """

    #Rechunk data in one lat and lon dim
    q = q.chunk({"lat":-1,"lon":-1})
    u = u.chunk({"lat":-1,"lon":-1})
    v = v.chunk({"lat":-1,"lon":-1})

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

    F = 0.5 * mag_dq * (tot_def * np.cos(2 * beta) - div) * 1.08e9

    out = xr.Dataset({"F":F})
    out["F"] = out["F"].assign_attrs(
        units = "g/kg/100km/3hr",
        long_name = "Moisture frontogenesis",
        description = "2d kinematic moisture frontogenesis parameter.")  

    return xr.Dataset({"F":F})

def coast_relative_frontogenesis(q,u,v,angle_da):

    """
    This function calculates 2d shearing and confluence in the moisture field, using a wind field that has been rotated to be coastline-relative.

    ## Input:
    * q: xarray dataarray of specific humidity in kg/kg

    * u: xarray dataarray of u winds in m/s

    * v: xarray dataarray of v winds in m/s

    * angle_da: xarray dataset containing coastline angles ("angle_interp")

    ## Output:
    * xarray dataset with variables of "shearing" and "confluence". These describe the changes in the moisture gradient due to strecthing and shearing relative to the coastline.
    
    """

    #Rechunk data in one lat and lon dim
    q = q.chunk({"lat":-1,"lon":-1})
    u = u.chunk({"lat":-1,"lon":-1})
    v = v.chunk({"lat":-1,"lon":-1})

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
    theta=angle_da 

    #Rotate angle to be perpendicular to theta, from E (i.e. mathamatical angle definition)
    rotated_angle=(((theta)%360-90)%360) + 90   

    #Define normal angle vectors, pointing onshore
    cx, cy = [-np.cos(np.deg2rad(rotated_angle)), np.sin(np.deg2rad(rotated_angle))]

    #Define normal angle vectors, pointing alongshore
    ax, ay = [-np.cos(np.deg2rad(rotated_angle - 90)), np.sin(np.deg2rad(rotated_angle - 90))]    

    #Calculate the wind component perpendicular (vprime) and parallel (uprime) to the coast by using the normal unit vectors
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
    out = xr.Dataset({
        "Fc":xr.DataArray((confluence+shearing) * 1.08e9,coords=q.coords)
        })
    out["Fc"] = out["Fc"].assign_attrs(
        units = "g/kg/100km/3hr",
        long_name = "Coastline-relative moisture frontogenesis",
        description = "2d kinematic moisture frontogenesis parameter calculated using coastline-relative winds and specific humidity.")  

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