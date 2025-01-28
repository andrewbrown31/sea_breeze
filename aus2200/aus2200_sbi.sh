#!/bin/bash

#PBS -P gb02 
#PBS -q hugemem
#PBS -l walltime=12:00:00,mem=1470GB 
#PBS -l ncpus=48
#PBS -l jobfs=190gb
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/aus2200_sbi.o 
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/aus2200_sbi.e
#PBS -l storage=gdata/gb02+gdata/hh5+gdata/ua8

export HDF5_USE_FILE_LOCKING=FALSE
 
#Set up conda/shell environments 
module use /g/data/hh5/public/modules
module load conda/analysis3
module load dask-optimiser

python /home/548/ab4502/working/sea_breeze/aus2200/aus2200_sbi.py "2016-01-01 01:00" "2016-01-31 23:00" --lat1 "-45.7" --lat2 "-6.9" --lon1 "108" --lon2 "158.5" --lev_chunk "-1" --lon_chunk "0" --lat_chunk "0" --time_chunk "1" --model aus2200 --exp_id "mjo-elnino" --hgt1 "0" --hgt2 "4500"
