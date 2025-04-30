#!/bin/bash

#PBS -P ng72 
#PBS -q normal
#PBS -l walltime=01:00:00,mem=128GB 
#PBS -l ncpus=48
#PBS -l jobfs=32gb
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/barra_c_smooth_s2_temporal_sb.o 
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/barra_c_smooth_s2_temporal_sb.e
#PBS -l storage=gdata/ng72+gdata/hh5+gdata/ua8+gdata/rt52+gdata/ob53+gdata/xp65
 
#Set up conda/shell environments 
module use /g/data/xp65/public/modules
module load conda/analysis3-24.07
module use /g/data/hh5/public/modules
module load dask-optimiser


python /home/548/ab4502/working/sea_breeze/barra_c/barra_c_temporal_sb.py "2013-01-01 00:00" "2013-01-31 23:00" --model barra_c_smooth_s2 --smooth --sigma 2
python /home/548/ab4502/working/sea_breeze/barra_c/barra_c_temporal_sb.py "2013-02-01 00:00" "2013-02-28 23:00" --model barra_c_smooth_s2 --smooth --sigma 2
python /home/548/ab4502/working/sea_breeze/barra_c/barra_c_temporal_sb.py "2016-01-01 00:00" "2016-01-31 23:00" --model barra_c_smooth_s2 --smooth --sigma 2
python /home/548/ab4502/working/sea_breeze/barra_c/barra_c_temporal_sb.py "2016-02-01 00:00" "2016-02-29 23:00" --model barra_c_smooth_s2 --smooth --sigma 2
python /home/548/ab4502/working/sea_breeze/barra_c/barra_c_temporal_sb.py "2018-01-01 00:00" "2018-01-31 23:00" --model barra_c_smooth_s2 --smooth --sigma 2
python /home/548/ab4502/working/sea_breeze/barra_c/barra_c_temporal_sb.py "2018-02-01 00:00" "2018-02-28 23:00" --model barra_c_smooth_s2 --smooth --sigma 2
