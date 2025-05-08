#!/bin/bash

#PBS -P ng72 
#PBS -q normal
#PBS -l walltime=0:10:00,mem=190GB 
#PBS -l ncpus=96
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/new_diagnostic_submission2.o 
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/new_diagnostic_submission2.e
#PBS -l storage=gdata/ng72+gdata/hh5+gdata/ua8+gdata/bs94+gdata/dk92+gdata/xp65

#Set up conda/shell environments 
module use /g/data/dk92/apps/Modules/modulefiles
module load gadi_jupyterlab/23.02

module use /g/data/xp65/public/modules
module load conda/analysis3-24.07

module use /g/data/hh5/public/modules
module load dask-optimiser
#module load conda/analysis3


#source activate analysis3
jupyter.ini.sh -D

python /home/548/ab4502/working/sea_breeze/test_notebooks/test_dask_jobqueue/new_diagnostic_submission.py
