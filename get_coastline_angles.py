from coastline_funcs import get_coastline_angle_kernel
from load_model_data import load_barra_static, load_era5_static, remove_era5_inland_lakes, load_aus2200_static

if __name__ == "__main__":

    #This slice is the outer bounds of the BARRA-C2 domain
    # lon_slice = slice(108,160)
    # lat_slice = slice(-45.7,-5.0)

    #This slice is a test slice for tasmania and the bottom part of the Aus mainland
    # lon_slice = slice(142,149)
    # lat_slice = slice(-45,-38)

    #This slice is the outer bounds of the AUS2200 domain
    lon_slice = slice(108,159)
    lat_slice = slice(-45.7,-6.831799)    

    #BARRA-R
    # _, barra_r_lsm = load_barra_static("AUS-11",lon_slice,lat_slice)
    # barra_r_angles = get_coastline_angle_kernel(barra_r_lsm)
    # barra_r_angles.to_netcdf("/g/data/gb02/ab4502/coastline_data/barra_r_angles_v2.nc")
    
    # #ERA5
    # _, era5_lsm, era5_cl = load_era5_static(lon_slice,lat_slice,"2023-01-01 00:00","2023-01-01 00:00")
    # era5_angles = get_coastline_angle_kernel(remove_era5_inland_lakes(era5_lsm,era5_cl))
    # era5_angles.to_netcdf("/g/data/gb02/ab4502/coastline_data/era5_angles_v2.nc") 

    # #BARRA-C
    # _, barra_c_lsm = load_barra_static("AUST-04",lon_slice,lat_slice)
    # barra_c_angles = get_coastline_angle_kernel(barra_c_lsm)
    # barra_c_angles.to_netcdf("/g/data/gb02/ab4502/coastline_data/barra_c_angles_v2.nc")       

    #ERA5 global
    # _, era5_lsm, era5_cl = load_era5_static(slice(0,360),slice(-90,90),"2023-01-01 00:00","2023-01-01 00:00")
    # era5_lsm["lon"] = ((era5_lsm.lon - 180) % 360) - 180
    # era5_lsm = era5_lsm.sortby("lon")
    # era5_cl["lon"] = ((era5_cl.lon - 180) % 360) - 180
    # era5_cl = era5_cl.sortby("lon")
    # era5_angles = get_coastline_angle_kernel(remove_era5_inland_lakes(era5_lsm,era5_cl))
    # era5_angles.to_netcdf("/g/data/gb02/ab4502/coastline_data/era5_global_angles_v2.nc")

    #AUS2200
    _, aus2200_lsm = load_aus2200_static("mjo-neutral",lon_slice,lat_slice)
    aus2200_angles = get_coastline_angle_kernel(aus2200_lsm)
    aus2200_angles.to_netcdf("/g/data/gb02/ab4502/coastline_data/aus2200_v2.nc")