#!/bin/bash

#PBS -P ng72 
#PBS -q hugemem
#PBS -l walltime=01:00:00,mem=512GB 
#PBS -l ncpus=48
#PBS -l jobfs=32gb
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/aus2200_fuzzy.o 
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/aus2200_fuzzy.e
#PBS -l storage=gdata/ng72+gdata/hh5+gdata/ua8
 
#Set up conda/shell environments 
module use /g/data/hh5/public/modules
module load conda/analysis3
module load dask-optimiser

python /home/548/ab4502/working/sea_breeze/aus2200/aus2200_fuzzy.py --model aus2200_smooth_s4
