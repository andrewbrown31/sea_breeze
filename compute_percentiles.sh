#!/bin/bash

#PBS -P ng72
#PBS -q hugemem
#PBS -l walltime=01:00:00,mem=512GB 
#PBS -l ncpus=48
#PBS -l jobfs=190gb
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/compute_percentiles.o 
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/compute_percentiles.e
#PBS -l storage=gdata/ng72+gdata/hh5+gdata/xp65+gdata/dk92
 
#Set up conda/shell environments 
module use /g/data/xp65/public/modules
module load conda/analysis3-24.07

module use /g/data/dk92/apps/Modules/modulefiles
module load gadi_jupyterlab/23.02
jupyter.ini.sh -D

python /home/548/ab4502/working/sea_breeze/compute_percentiles.py
