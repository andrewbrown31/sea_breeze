#!/bin/bash

#PBS -P ng72 
#PBS -q normal
#PBS -l walltime=01:00:00,mem=64GB 
#PBS -l ncpus=48
#PBS -l jobfs=32gb
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/era5_filter.o 
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/era5_filter.e
#PBS -l storage=gdata/ng72+gdata/hh5+gdata/ua8+gdata/rt52+gdata/ob53
 
#Set up conda/shell environments 
module use /g/data/hh5/public/modules
module load conda/analysis3
module load dask-optimiser

python /home/548/ab4502/working/sea_breeze/era5/filter.py --model era5 --filter_name no_hourly_change
