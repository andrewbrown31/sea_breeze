#!/bin/bash

#PBS -P gb02 
#PBS -q normal
#PBS -l walltime=12:00:00,mem=128GB 
#PBS -l ncpus=48
#PBS -l jobfs=32gb
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/aus2200_get_coastline_angles.o 
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/aus2200_get_coastline_angles.e
#PBS -l storage=gdata/gb02+gdata/ob53+gdata/hh5+gdata/rt52+gdata/ua8
 
#Set up conda/shell environments 
module use /g/data/hh5/public/modules
module load conda/analysis3
module load dask-optimiser

python /home/548/ab4502/working/sea_breeze/aus2200/get_coastline_angles.py
