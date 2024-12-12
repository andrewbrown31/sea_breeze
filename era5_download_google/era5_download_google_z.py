import xarray as xr
import pandas as pd

#Load in ERA5 model level zarr from Google cloud
#From here: https://github.com/google-research/arco-era5
ds = xr.open_zarr(
    'gs://gcp-public-data-arco-era5/ar/model-level-1h-0p25deg.zarr-v1',
    storage_options=dict(token='anon'),
    chunks={"time":1,"hybrid":-1}
)

#Slice down a bit
t1 = "2016-01-01 00:00"
t2 = "2016-01-31 23:00"
ds_sliced = ds.sel(
    time=slice(t1,t2),
    latitude=slice(-5,-45.7),
    longitude=slice(108,160),
    hybrid=slice(100,137)
)["geopotential"].chunk({"time":48})

fname = "era5_mdl_z_"+pd.to_datetime(t1).strftime("%Y%m%d")+"_"+pd.to_datetime(t2).strftime("%Y%m%d")+".nc"
complevel = 9
encoding_dict = {"geopotential":{"zlib": True, "complevel": complevel}}
ds_sliced.to_netcdf("/g/data/gb02/ab4502/era5_model_lvl/"+fname,encoding=encoding_dict)
