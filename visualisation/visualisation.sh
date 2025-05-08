#!/bin/bash

#PBS -P ng72
#PBS -q normal
#PBS -l walltime=01:00:00,mem=190GB 
#PBS -l ncpus=48
#PBS -l jobfs=32gb
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/vis.o
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/vis.e
#PBS -l storage=gdata/hh5+gdata/bs94+gdata/ng72+scratch/ng72
 
#Set up conda/shell environments 
module use /g/data/hh5/public/modules
module load conda/analysis3
module load dask-optimiser

python /home/548/ab4502/working/sea_breeze/visualisation/visualisation.py