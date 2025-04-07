from sea_breeze.era5_download_google.era5_download_google import download_era5_google

if __name__ == "__main__":

    t1s = ["2013-01-01 00:00","2013-02-01 00:00","2016-02-01 00:00","2018-01-01 00:00","2018-02-01 00:00"]
    t2s = ["2013-01-31 23:00","2013-02-28 23:00","2016-02-29 23:00","2018-01-31 23:00","2018-02-28 23:00"]
    v = "v_component_of_wind"

    for t1,t2 in zip(t1s,t2s):
        print(f"Downloading {v} from {t1} to {t2}")
        download_era5_google(t1,t2,v)