#So far this code has been used in ARE

import dask.config
from dask.distributed import Client,LocalCluster
from dask_jobqueue import PBSCluster

cluster = PBSCluster(
    cores=48,
    processes=48,
    memory="190GB",
    walltime="01:00:00",
    local_directory='$TMPDIR',
    header_skip=["select"],
    job_extra_directives=['-P ng72','-l mem=190GB','-l ncpus=48','-l storage=gdata/xp65+gdata/ng72+gdata/hh5+gdata/ua8+gdata/bs94'],
    python="/g/data/xp65/public/apps/med_conda_scripts/analysis3-24.07.d/bin/python")

cluster.scale(jobs=2)

client = Client(cluster)


#ARE SETTINGS

#/g/data3/xp65/public/modules /g/data/hh5/public/modules
#conda/analysis3-25.04 dask-optimiser