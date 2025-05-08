#!/bin/bash

#PBS -P ng72 
#PBS -q megamem
#PBS -l walltime=6:00:00,mem=2940GB 
#PBS -l ncpus=48
#PBS -l jobfs=190gb
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/aus2200_smooth6_sbi_2016.o 
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/aus2200_smooth6_sbi_2016.e
#PBS -l storage=gdata/ng72+gdata/hh5+gdata/ua8+gdata/bs94+gdata/xp65
 
#Set up conda/shell environments 
#module use /g/data/xp65/public/modules
#module load conda/analysis3-24.07
module use /g/data/hh5/public/modules
module load dask-optimiser
module load conda/analysis3

python /home/548/ab4502/working/sea_breeze/aus2200/aus2200_sbi.py "2016-01-01 01:00" "2016-01-31 23:00" --lat1 "-45.7" --lat2 "-6.9" --lon1 "108" --lon2 "158.5" --lev_chunk "-1" --lon_chunk "-1" --lat_chunk "-1" --time_chunk "1" --model aus2200_smooth_s6 --exp_id "mjo-elnino2016" --hgt1 "0" --hgt2 "4500" --interp_hgt --smooth --sigma 6

python /home/548/ab4502/working/sea_breeze/aus2200/aus2200_sbi.py "2016-02-01 01:00" "2016-02-29 23:00" --lat1 "-45.7" --lat2 "-6.9" --lon1 "108" --lon2 "158.5" --lev_chunk "-1" --lon_chunk "-1" --lat_chunk "-1" --time_chunk "1" --model aus2200_smooth_s6 --exp_id "mjo-elnino2016" --hgt1 "0" --hgt2 "4500" --interp_hgt --smooth --sigma 6