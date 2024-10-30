#!/bin/bash

#PBS -P gb02 
#PBS -q hugemem
#PBS -l walltime=12:00:00,mem=1470GB 
#PBS -l ncpus=1
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/get_coastline_angles.o 
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/get_coastline_angles.e
#PBS -l storage=gdata/gb02+gdata/ob53+gdata/hh5+gdata/rt52
 
#Set up conda/shell environments 
module use /g/data/hh5/public/modules
module load conda/analysis3

python /home/548/ab4502/working/sea_breeze/get_coastline_angles.py