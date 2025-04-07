#!/bin/bash

#PBS -P ng72 
#PBS -q hugemem
#PBS -l walltime=06:00:00,mem=1470GB 
#PBS -l ncpus=48
#PBS -l jobfs=32gb
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/aus2200_smooth_s2_filter.o 
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/aus2200_smooth_s2_filter.e
#PBS -l storage=gdata/hh5+gdata/bs94+gdata/ng72
 
#Set up conda/shell environments 
module use /g/data/hh5/public/modules
module load conda/analysis3
module load dask-optimiser

python /home/548/ab4502/working/sea_breeze/aus2200/filter.py --model aus2200_smooth_s2 --filter_name no_hourly_change
