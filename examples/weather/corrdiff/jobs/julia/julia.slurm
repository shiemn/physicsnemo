#!/bin/bash
#SBATCH -J PrecipitationDownscaling
#SBATCH -c 16
#SBATCH -p standard
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=simon.hentschel@uni-wuerzburg.de
#SBATCH --output=output/job-%j.out
#SBATCH --tmp=1000G



export WANDB__SERVICE_WAIT=300
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
echo ${CUDA_VISIBLE_DEVICES}

echo "Copying HCLIM3 to scratch"
#measure runtime
start=$(date +%s.%N)
mkdir -p /tmp/Norway/HCLIM3/preprocessed_large/


cp -n -r /home/sih25nq/data/downscaling/Norway/HCLIM3/preprocessed_large/NorCP_AROME_EC-EARTH /tmp/Norway/HCLIM3/preprocessed_large/. &
cp /home/sih25nq/data/downscaling/Norway/HCLIM3/orog_NEU-3_ECMWF-ERAINT_evaluation_r1i1p1_HCLIMcom-HCLIM38-AROME_x2yn2v1_fx.nc /tmp/Norway/HCLIM3/.
cp /home/sih25nq/data/downscaling/Norway/HCLIM3/orog_NEU-3_ECMWF-ERAINT_evaluation_r1i1p1_HCLIMcom-HCLIM38-AROME_x2yn2v1_fx.nc /tmp/Norway/HCLIM3/preprocessed_large/.

wait


end=$(date +%s.%N)
echo "Copying data to scratch took $(awk "BEGIN {print ${end} - ${start}}") seconds"
echo "Copying data to scratch complete"



apptainer run --nv --bind /tmp/:/data --bind ~/bjerknes/corrdiff/checkpoints:/checkpoints/ ~/new_corrdiff.sif torchrun --nproc_per_node=1 ~/bjerknes/corrdiff/train.py --config-name $1 \