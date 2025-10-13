#!/bin/bash
#SBATCH -J PrecipitationDownscalingParallel
#SBATCH -c 16
#SBATCH -p h100
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=simon.hentschel@uni-wuerzburg.de
#SBATCH --output=output/job-%j.out
#SBATCH --error=output/job-%j.err
#SBATCH --tmp=1000G



export WANDB__SERVICE_WAIT=300
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
echo ${CUDA_VISIBLE_DEVICES}

echo "Copying HCLIM3 to scratch"
#measure runtime
start=$(date +%s.%N)
mkdir -p /tmp/Norway/HCLIM3/preprocessed/


cp -n -r ~/datatemp/Norway/HCLIM3/preprocessed/NorCP_AROME_EC-EARTH /tmp/Norway/HCLIM3/preprocessed/NorCP_AROME_EC-EARTH &
cp ~/datatemp/Norway/HCLIM3/orog_NEU-3_ECMWF-ERAINT_evaluation_r1i1p1_HCLIMcom-HCLIM38-AROME_x2yn2v1_fx.nc /tmp/Norway/HCLIM3/.

wait


end=$(date +%s.%N)
echo "Copying data to scratch took $(awk "BEGIN {print ${end} - ${start}}") seconds"
echo "Copying data to scratch complete"



apptainer run --nv --bind /tmp/:/data --bind ~/bjerknes/corrdiff/checkpoints:/checkpoints/ ~/own_corrdiff.sif torchrun --nproc_per_node=8  ~/bjerknes/corrdiff/train.py --config-name $1 \

#apptainer run --nv --bind ~/data/downscaling:/data --bind ~/bjerknes/corrdiff/checkpoints:/checkpoints/ ~/own_corrdiff.sif torchrun --nproc_per_node=4  ~/bjerknes/corrdiff/train.py --config-name $1 \

