import xarray as xr
import metpy
import metpy.calc as mpcalc

def load_half_hourly_stn_obs(state,time_slice):

    '''
    Load half-hourly AWS data and slice based on time. Also convert wind speed and direction to 
    u and v wind components

    state = str, one of "NSW-ACT", "NT", "QLD", "SA", "TAS-ANT", "VIC", "WA"
    time_slice = slice of strings e.g. slice("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M")
    '''

    path = "/g/data/w40/clv563/BoM_data_202409/half_hourly_data_netcdf/"
    stn_obs = xr.open_dataset(path + "AWS-data-" + state + ".nc").sel(time=time_slice)
    u,v = metpy.calc.wind_components(
        stn_obs.wspd.metpy.convert_units("m/s"),
        stn_obs.wdir * metpy.units.units.deg)
    stn_obs["u"] = u
    stn_obs["v"] = v

    #Calculate specific humidity. TODO: Change from mlsp to sp.
    stn_obs["Tdew"] = stn_obs["Tdew"].assign_attrs(units = "degC")
    stn_obs["hus"] = mpcalc.specific_humidity_from_dewpoint(stn_obs["mslp"],stn_obs["Tdew"])

    return stn_obs