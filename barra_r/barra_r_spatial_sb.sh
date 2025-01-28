#!/bin/bash

#PBS -P gb02 
#PBS -q normal
#PBS -l walltime=1:00:00,mem=64GB 
#PBS -l ncpus=48
#PBS -l jobfs=32gb
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/barra_r_spatial_sb.o 
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/barra_r_spatial_sb.e
#PBS -l storage=gdata/gb02+gdata/hh5+gdata/ua8+gdata/rt52+gdata/ob53
 
export HDF5_USE_FILE_LOCKING=FALSE

#Set up conda/shell environments 
module use /g/data/hh5/public/modules
module load conda/analysis3
module load dask-optimiser

python /home/548/ab4502/working/sea_breeze/barra_r/barra_r_spatial_sb.py "2016-01-01 00:00" "2016-01-31 23:00"