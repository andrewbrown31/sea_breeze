#!/bin/bash

#PBS -P ng72 
#PBS -q hugemem
#PBS -l walltime=6:00:00,mem=512GB 
#PBS -l ncpus=48
#PBS -l jobfs=32gb
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/aus2200_spatial_sb.o 
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/aus2200_spatial_sb.e
#PBS -l storage=gdata/gb02+gdata/hh5+gdata/ua8
 
#Set up conda/shell environments 
module use /g/data/hh5/public/modules
module load conda/analysis3
module load dask-optimiser

python /home/548/ab4502/working/sea_breeze/aus2200/aus2200_spatial_sb.py "2016-01-01 00:00" "2016-01-31 23:00" --lon_chunk "-1" --lat_chunk "-1" --time_chunk "1" --model aus2200_smooth_s4 --exp_id "mjo-elnino" --smooth --sigma 4
