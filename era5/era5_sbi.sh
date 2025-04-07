#!/bin/bash

#PBS -P ng72 
#PBS -q normal
#PBS -l walltime=2:00:00,mem=64GB 
#PBS -l ncpus=48
#PBS -l jobfs=1gb
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/era5_sbi.o 
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/era5_sbi.e
#PBS -l storage=gdata/ng72+gdata/hh5+gdata/ua8+gdata/rt52

#Set up conda/shell environments 
module use /g/data/hh5/public/modules
module load conda/analysis3
module load dask-optimiser

python /home/548/ab4502/working/sea_breeze/era5/era5_sbi.py "2013-01-01 00:00" "2013-01-31 23:00" --lat1 "-45.7" --lat2 "-6.9" --lon1 "108" --lon2 "158.5" --model era5 --hgt1 "0" --hgt2 "4500"
python /home/548/ab4502/working/sea_breeze/era5/era5_sbi.py "2013-02-01 00:00" "2013-02-28 23:00" --lat1 "-45.7" --lat2 "-6.9" --lon1 "108" --lon2 "158.5" --model era5 --hgt1 "0" --hgt2 "4500"
python /home/548/ab4502/working/sea_breeze/era5/era5_sbi.py "2016-01-01 00:00" "2016-01-31 23:00" --lat1 "-45.7" --lat2 "-6.9" --lon1 "108" --lon2 "158.5" --model era5 --hgt1 "0" --hgt2 "4500"
python /home/548/ab4502/working/sea_breeze/era5/era5_sbi.py "2016-02-01 00:00" "2016-02-29 23:00" --lat1 "-45.7" --lat2 "-6.9" --lon1 "108" --lon2 "158.5" --model era5 --hgt1 "0" --hgt2 "4500"
python /home/548/ab4502/working/sea_breeze/era5/era5_sbi.py "2018-01-01 00:00" "2018-01-31 23:00" --lat1 "-45.7" --lat2 "-6.9" --lon1 "108" --lon2 "158.5" --model era5 --hgt1 "0" --hgt2 "4500"
python /home/548/ab4502/working/sea_breeze/era5/era5_sbi.py "2018-02-01 00:00" "2018-02-28 23:00" --lat1 "-45.7" --lat2 "-6.9" --lon1 "108" --lon2 "158.5" --model era5 --hgt1 "0" --hgt2 "4500"
