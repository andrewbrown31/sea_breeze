#!/bin/bash

#PBS -P ng72 
#PBS -q copyq
#PBS -l walltime=10:00:00,mem=16GB 
#PBS -l ncpus=1
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/era5_download_google_u.o 
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/era5_download_google_u.e
#PBS -l storage=gdata/ng72+gdata/hh5
 
#Set up conda/shell environments 
module use /g/data/hh5/public/modules
module load conda/analysis3

python /home/548/ab4502/working/sea_breeze/era5_download_google/era5_download_google_u.py
