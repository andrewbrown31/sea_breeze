import xarray as xr
import metpy
import metpy.calc as mpcalc
import numpy as np
import datetime as dt
import zipfile
import glob

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

def unpack_level1b(rid, times):
	
	"""
	Unzip level1b radar data from the rq0 project between times[0] and times[1], and save to scratch
	
	## INPUTS
	* rid: str, radar id
	* times: list of datetime objects, [start, end]
	
	"""
	assert times[0].year == times[1].year, "Times range must be within calendar year"
	files = np.array(glob.glob("/g/data/rq0/level_1b/"+rid+"/grid/"+str(times[0].year)+"/*.zip"))
	if len(files) == 0:
		print("NO FILES FOUND FOR RID: "+rid+" AND TIMES "+str(times[0])+" "+str(times[-1]))
	file_dates = np.array([dt.datetime.strptime(f.split("/")[8].split("_")[1], "%Y%m%d") for f in files])
	target_files = files[(file_dates >= times[0].replace(hour=0, minute=0)) & (file_dates <= times[1].replace(hour=0, minute=0))]
	extract_to = "/scratch/gb02/ab4502/radar/"
	for f in target_files:
		with zipfile.ZipFile(f, "r") as zip_ref:
			zip_ref.extractall(extract_to)
			
def load_radar_level1b(rid,times):
	
    """
	Load radar data from the rq0 project between times[0] and times[1] as an xarray dataset
	
	## INPUTS
	* rid: str, radar id
	* times: list of datetime objects, [start, end]
	
	## OUTPUTS
	* radar_ds: xarray dataset, radar data
	"""
	
    #Unpack the zipped rq0 radar files
    unpack_level1b(rid, times)
	
    #Constuct a list of the unpacked grid files
    grid_files = np.sort(glob.glob("/scratch/gb02/ab4502/radar/"+rid+"*_grid.nc"))
	
    #Select the files that are within the time range
    file_dates = np.array([dt.datetime.strptime(f.split("/")[5].split("_")[1] + f.split("/")[5].split("_")[2],\
                    "%Y%m%d%H%M%S") for f in grid_files])
	
    target_files = grid_files[(file_dates >= times[0]) & (file_dates <= times[1])]
	
    #Open the files using xarray
    radar_ds = xr.open_mfdataset(target_files)

    return radar_ds