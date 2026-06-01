#!/bin/bash
# Copy Taiwan CWA dataset to node-local scratch
# Called via: srun --ntasks-per-node=1 bash copy_taiwan.sh
# srun acts as a barrier - it waits for all nodes to complete before returning

TAR_SRC=/hnvme/workspace/b214cb11-helma-ecodata/downscaling/CorrDiff/cwa_dataset.tar

echo "Node ${SLURM_NODEID:-0}: Extracting CWA zarr from tar to /scratch..."
start=$(date +%s.%N)

tar -xf $TAR_SRC -C /scratch/ cwa_dataset/cwa_dataset.zarr

end=$(date +%s.%N)
echo "Node ${SLURM_NODEID:-0}: Extraction took $(awk "BEGIN {print ${end} - ${start}}") seconds"
echo "Node ${SLURM_NODEID:-0}: Done"
