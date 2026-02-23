#!/bin/bash
#SBATCH -J PrecipitationDownscaling
#SBATCH -c 16
#SBATCH -p standard
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=simon.hentschel@uni-wuerzburg.de
#SBATCH --output=output/job-%j.out
#SBATCH --tmp=1000G



export APPTAINERENV_HYDRA_FULL_ERROR=1
export APPTAINERENV_OMP_NUM_THREADS=16
export APPTAINERENV_PYTHONNOUSERSITE=1


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export WANDB__SERVICE_WAIT=300
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
echo ${CUDA_VISIBLE_DEVICES}

echo "Copying HCLIM3 to scratch"
#measure runtime
start=$(date +%s.%N)
mkdir -p /tmp/cwb/


cp -n -r /data/42-julia-hpc-rz-lsx/sih25nq/downscaling/CorrDiff/cwa_dataset/cwa_dataset_3months.zarr /tmp/cwb/. 

wait


end=$(date +%s.%N)
echo "Copying data to scratch took $(awk "BEGIN {print ${end} - ${start}}") seconds"
echo "Copying data to scratch complete"



apptainer run --nv --bind /tmp/:/data --bind ~/bjerknes/corrdiff/checkpoints:/checkpoints/ ~/new_corrdiff.sif torchrun --nproc_per_node=1 ~/bjerknes/corrdiff/train.py --config-name $1 

