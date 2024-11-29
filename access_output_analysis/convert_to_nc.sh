#!/bin/bash

module use /g/data/hh5/public/modules
module load conda/analysis3

cd /scratch/gb02/ab4502/cylc-run/u-dg768/share/cycle/20240129T0000Z/Gippsland/d0198/RAL3P2/um/
python /g/data/hh5/public/apps/miniconda3/envs/analysis3-24.04/lib/python3.10/site-packages/amami/um2nc.py umnsaa_pvera000