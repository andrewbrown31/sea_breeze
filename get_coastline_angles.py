from coastline_funcs import get_coastline_angle_kernel, fill_coastline_angles
from load_model_data import load_barra_static, load_era5_static, remove_era5_inland_lakes

if __name__ == "__main__":

    lon_slice = slice(108,160)
    lat_slice = slice(-45.7,-5.0)

    #BARRA-R
    _, barra_r_lsm = load_barra_static("AUS-11",lon_slice,lat_slice)
    barra_r_angles = get_coastline_angle_kernel(barra_r_lsm)
    fill_coastline_angles(barra_r_angles).to_netcdf("/g/data/gb02/ab4502/coastline_data/barra_r_angles.nc")
    
    #ERA5
    _, era5_lsm = load_era5_static(lon_slice,lat_slice,"2024-01-01 00:00","2024-01-01 00:00")
    era5_angles = get_coastline_angle_kernel(remove_era5_inland_lakes(era5_lsm))
    fill_coastline_angles(era5_angles).to_netcdf("/g/data/gb02/ab4502/coastline_data/era5_angles.nc") 

    #BARRA-C
    _, barra_c_lsm = load_barra_static("AUST-04",lon_slice,lat_slice)
    barra_c_angles = get_coastline_angle_kernel(barra_c_lsm)
    fill_coastline_angles(barra_c_angles).to_netcdf("/g/data/gb02/ab4502/coastline_data/barra_c_angles.nc")       

    #ERA5 global
    _, era5_lsm = load_era5_static(slice(0,360),slice(-90,90),"2024-01-01 00:00","2024-01-01 00:00")
    era5_lsm["lon"] = ((era5_lsm.lon - 180) % 360) - 180
    era5_lsm = era5_lsm.sortby("lon")
    era5_angles = get_coastline_angle_kernel(remove_era5_inland_lakes(era5_lsm))
    fill_coastline_angles(era5_angles).to_netcdf("/g/data/gb02/ab4502/coastline_data/era5_global_angles.nc")