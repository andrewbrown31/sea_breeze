#!/bin/bash

#PBS -P ng72 
#PBS -q hugemem
#PBS -l walltime=01:00:00,mem=512GB 
#PBS -l ncpus=48
#PBS -l jobfs=32gb
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/barra_c_fuzzy.o 
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/barra_c_fuzzy.e
#PBS -l storage=gdata/ng72+gdata/hh5+gdata/ua8
 
#Set up conda/shell environments 
module use /g/data/hh5/public/modules
module load conda/analysis3
module load dask-optimiser

python /home/548/ab4502/working/sea_breeze/barra_c/barra_c_fuzzy.py --model barra_c_smooth_s2
