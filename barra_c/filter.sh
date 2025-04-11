#!/bin/bash

#PBS -P gb02 
#PBS -q hugemem
#PBS -l walltime=12:00:00,mem=1470GB 
#PBS -l ncpus=48
#PBS -l jobfs=32gb
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/barra_c_filter.o 
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/barra_c_filter.e
#PBS -l storage=gdata/gb02+gdata/hh5+gdata/ua8+gdata/rt52+gdata/ob53
 
#Set up conda/shell environments 
module use /g/data/hh5/public/modules
module load conda/analysis3
module load dask-optimiser

#Threshold settings for BARRA-C
fc_threshold="14.328516"
f_threshold="16.098763"	
fuzzy_threshold="0.297940"

#Set the start and end dates, and the current date as the start date
start_date="2013-01-01"
end_date="2018-02-28"
current_date="$start_date 00:00:00"

#Set an optional interval for the time range
# This is the interval that will be used to calculate the end time for each iteration
#For monthly data, this should be 31 days or greater. It is a maximum of monthly intervals
optional_interval="10 days"

while [[ "$current_date" < "$end_date" ]]; do
    
    #Create a datetime object for the current date
    start_time=$(date -d "$current_date" +"%Y-%m-%d %H:%M")

    #Calculate the current month and next month
    start_month=$(date -d "$current_date" +"%Y-%m-01")
    next_month=$(date -d "$start_month +1 month" +"%Y-%m-01")

    #Set the end time based on the specified interval, minus one hour
    end_time=$(date -ud "$start_time UTC + $optional_interval" +"%Y-%m-%d %H:%M")
    end_time=$(date -ud "$end_time UTC - 1 hour" +"%Y-%m-%d %H:%M")

    #If the end time is greater than the next month, set the end time to the last hour of the month
    if [[ $(date -d "$end_time" +"%s") -gt $(date -d "$next_month" +"%s") ]]; then
        end_time=$(date -d "$next_month -1 hour" +"%Y-%m-%d %H:%M")
        new_month=1
    fi

    #We only have data for Jan+Feb for 2013, 2016 and 2018, so check we are in those ranges
    if [[ "$current_date" == *"-01-"* || "$current_date" == *"-02-"* ]]; then
        if [[ "$current_date" == *"2013"* || "$current_date" == *"2016"* || "$current_date" == *"2018"* ]]; then

            #Print the date interval
            echo "$start_time" "$end_time" 

            #Run the filter script for each field with the specified parameters

            #Fc     
            python /home/548/ab4502/working/sea_breeze/filter.py --model barra_c_smooth_s2 --filter_name no_hourly_change --field_name Fc --t1 "$start_time" --t2 "$end_time" --threshold fixed --threshold_value $fc_threshold

            #F
            python /home/548/ab4502/working/sea_breeze/filter.py --model barra_c_smooth_s2 --filter_name no_hourly_change --field_name F --t1 "$start_time" --t2 "$end_time" --threshold fixed --threshold_value $f_threshold

            #Fuzzy
            python /home/548/ab4502/working/sea_breeze/filter.py --model barra_c_smooth_s2 --filter_name no_hourly_change --field_name fuzzy --t1 "$start_time" --t2 "$end_time" --threshold fixed --threshold_value $fuzzy_threshold

        fi
    fi

    #If we are into the next month, advance to the first of the next month
    #Otherwise, advance by the specified interval
    if [[ "$new_month" == 1 ]]; then
        current_date=$(date -d "$next_month" +"%Y-%m-%d %H:%M")
        new_month=0
    else
        current_date=$(date -ud "$current_date UTC + $optional_interval" +"%Y-%m-%d %H:%M")
    fi

done