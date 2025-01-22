#!/bin/bash

#PBS -P gb02 
#PBS -q hugemem
#PBS -l walltime=06:00:00,mem=512GB 
#PBS -l ncpus=48
#PBS -l jobfs=32gb
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/barra_c_filter.o 
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/barra_c_filter.e
#PBS -l storage=gdata/gb02+gdata/hh5+gdata/ua8+gdata/rt52+gdata/ob53
 
#Set up conda/shell environments 
module use /g/data/hh5/public/modules
module load conda/analysis3
module load dask-optimiser

python /home/548/ab4502/working/sea_breeze/barra_c/filter.py
