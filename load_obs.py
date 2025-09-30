import xarray as xr
import metpy
import metpy.calc as mpcalc
import numpy as np
import datetime as dt
import zipfile
import glob
import pyproj
import cartopy.crs as ccrs
import pandas as pd
import tqdm

def load_half_hourly_stn_obs_txt(state):

    """
    Load half-hourly AWS data for a given state from text files and save as an xarray dataset.
    Just for surface pressure (sp) data.
    """

    print("Loading half-hourly AWS data for state: " + state)

    #Get the file paths for the half-hourly data text files
    file_paths = glob.glob("/g/data/w40/clv563/BoM_data_202409/half_hourly_data/"+state+"/HM01X_Data_??????_*.txt")

    #Initialize lists to store data
    f_list = []
    stn_num_list = []
    lati = []
    long = []
    name = []

    #Define timezones for each state
    tzs = {"WA":"Australia/Perth", "NT":"Australia/Darwin", "SA":"Australia/Adelaide",
    "QLD":"Australia/Brisbane","NSW-ACT":"Australia/Sydney", "VIC":"Australia/Melbourne",
    "TAS-ANT":"Australia/Hobart"}    

    #Loop through each file and read the data
    for file_path in tqdm.tqdm(file_paths):
        f = pd.read_csv(
            file_path,
            usecols=[1,2,3,4,5,6,14,18,22,24,30],
            names=["bmid","year","month","day","hour","minute","temp","Tdew","wspd","wdir","sp"],
            header=0,
            dtype={"sp":float,"wdir":float,"wspd":float,"Tdew":float,"temp":float},
            na_values=['      ','     ','   ','#####'])
        f["time"] = pd.to_datetime(f[["year","month","day","hour","minute"]])
        f = f.drop(columns=["year","month","day","minute","hour","bmid"])
        f = f.drop_duplicates(subset=["time"])

        #Convert to UTC. Remove ambiguous times by setting them to NaT
        f["time"] = f["time"].dt.tz_localize(tzs[state],ambiguous="NaT",nonexistent="NaT").dt.tz_convert("UTC").dt.tz_localize(None)
        f = f.loc[~f.time.isna()]

        #Convert to xarray dataset and set time as index
        f_list.append(f.set_index("time").to_xarray())

        #Keep track of station number, latitude, longitude, and name
        stn_num = int(file_path.split("/")[-1].split("_")[2])
        stn_num_list.append(stn_num)
        stn_info = pd.read_csv(glob.glob("/g/data/w40/clv563/BoM_data_202409/half_hourly_data/"+state+"/HM01X_StnDet_*.txt")[0],header=None)
        lati.append(stn_info[stn_info.iloc[:,1] == stn_num][6].values[0])
        long.append(stn_info[stn_info.iloc[:,1] == stn_num][7].values[0])
        name.append(stn_info[stn_info.iloc[:,1] == stn_num][3].values[0])

    #Concatenate the list of xarray datasets into a single dataset
    stn_xr = xr.concat(f_list,dim="station",join="outer",fill_value=np.nan)

    #Add station metadata to the dataset, as well as latitude, longitude, and name
    stn_xr = stn_xr.merge(
        xr.Dataset(
            {"bmid":("station",stn_num_list),
             "latitude":("station",lati),
             "longitude":("station",long),
             "name":("station",name)},)
        )

    #Set attributes for the dataset
    stn_xr["sp"] = stn_xr.sp.assign_attrs({"long_name":"station_pressure","unit":"hPa"})
    stn_xr["temp"] = stn_xr.temp.assign_attrs({"long_name":"air_temperature","unit":"degC"})
    stn_xr["Tdew"] = stn_xr.Tdew.assign_attrs({"long_name":"dewpoint_temperature","unit":"degC"})
    stn_xr["wspd"] = stn_xr.wspd.assign_attrs({"long_name":"wind_speed","unit":"km/hr"})
    stn_xr["wdir"] = stn_xr.wdir.assign_attrs({"long_name":"wind_direction","unit":"degrees_from_north"})
    stn_xr["bmid"] = stn_xr.bmid.assign_attrs({"long_name":"Bureau of Meteorology Station ID"})

    #Save the dataset to a netCDF file
    stn_xr.to_netcdf("/g/data/ng72/ab4502/BoM_data_202409/half_hourly_data_netcdf/AWS-data-" + state + ".nc")

def load_half_hourly_stn_obs(state,time_slice):

    '''
    Load half-hourly AWS data and slice based on time. Also convert wind speed and direction to 
    u and v wind components

    Uses output fromload_half_hourly_stn_obs_txt

    state = str, one of "NSW-ACT", "NT", "QLD", "SA", "TAS-ANT", "VIC", "WA"
    time_slice = slice of strings e.g. slice("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M")
    '''

    #path = "/g/data/w40/clv563/BoM_data_202409/half_hourly_data_netcdf/"
    path = "/g/data/ng72/ab4502/BoM_data_202409/half_hourly_data_netcdf/"
    stn_obs = xr.open_dataset(path + "AWS-data-" + state + ".nc").sel(time=time_slice).set_index(station="bmid")
    stn_obs["Tdew"] = stn_obs["Tdew"].assign_attrs(units = "degC")
    stn_obs["sp"] = stn_obs["sp"].assign_attrs(units = "hPa")
    stn_obs["wspd"] = stn_obs["wspd"].assign_attrs(units = "km/hr")

    #Convert wind speed and direction to u and v components in m/s
    u,v = metpy.calc.wind_components(
        stn_obs.wspd.metpy.convert_units("m/s"),
        stn_obs.wdir * metpy.units.units.deg)
    stn_obs["u"] = u.pint.dequantify()
    stn_obs["v"] = v.pint.dequantify()

    #Calculate specific humidity.
    stn_obs["hus"] = mpcalc.specific_humidity_from_dewpoint(stn_obs["sp"],stn_obs["Tdew"])
    stn_obs["hus"] = stn_obs["hus"].pint.dequantify()

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

def load_hima(datetime,x_slice=slice(-3e6,2e6),y_slice=slice(-1e6,-4.5e6)):

    """
    Load Himawari data for a given time slice. The default x and y slices correspond to
    mainland Australia
    """

    #Load a single time slice of Himawari data
    ds = xr.open_mfdataset("/g/data/ra22/satellite-products/arc/obs/himawari-ahi/fldk/latest/"+\
                          datetime.strftime("%Y")+"/"+\
                          datetime.strftime("%m")+"/"+\
                          datetime.strftime("%d")+"/"+\
                          datetime.strftime("%H%M")+"/"+\
                          datetime.strftime("%Y%m%d%H%M%S")+\
                          "-P1S-ABOM_CREFL_B03-PRJ_GEOS141_500-HIMAWARI*-AHI.nc",chunks={"x":-1,"y":-1})

    #Define geostationary projection from file metadata projections firstly with cartopy then pyproj
    ccrs_proj = ccrs.Geostationary(central_longitude=ds["geostationary"].attrs["longitude_of_projection_origin"],satellite_height=35785863)
    pyproj_proj = pyproj.Proj(ccrs_proj)

    #Slice the dataset in x-y space. The defaults here roughly correspond to mainland Australia
    ds = ds.sel(x=x_slice,y=y_slice)
    
    return ds, pyproj_proj, ccrs_proj

def get_lat_lons(ds, proj):

    #Define x and y 2d coordinate arrays
    xx,yy=np.meshgrid(ds["x"],ds["y"])

    #Retrieve lats and lons from the projection
    lons, lats = proj(xx, yy, inverse=True)

    return lons, lats
