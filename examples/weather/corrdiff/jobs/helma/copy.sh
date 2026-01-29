#!/bin/bash
# Copy data to node-local scratch
# Called via: srun --ntasks-per-node=1 bash copy.sh
# srun acts as a barrier - it waits for all nodes to complete before returning

echo "Node ${SLURM_NODEID:-0}: Copying HCLIM3 to scratch"
start=$(date +%s.%N)

# Create directories
mkdir -p /scratch/Norway/HCLIM3/preprocessed_large/

# Copy data (each node copies to its own /scratch)
cp -n -r /hnvme/workspace/b214cb11-helma-ecodata/downscaling/Norway/HCLIM3/preprocessed_large/NorCP_AROME_EC-EARTH /scratch/Norway/HCLIM3/preprocessed_large/NorCP_AROME_EC-EARTH &
cp /hnvme/workspace/b214cb11-helma-ecodata/downscaling/Norway/HCLIM3/orog_NEU-3_ECMWF-ERAINT_evaluation_r1i1p1_HCLIMcom-HCLIM38-AROME_x2yn2v1_fx.nc /scratch/Norway/HCLIM3/.

wait

end=$(date +%s.%N)
echo "Node ${SLURM_NODEID:-0}: Copying data to scratch took $(awk "BEGIN {print ${end} - ${start}}") seconds"
echo "Node ${SLURM_NODEID:-0}: Copying data to scratch complete"
