#!/bin/bash
#SBATCH -J PrecipitationDownscalingParallel
#SBATCH -c 16
#SBATCH -p h100
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=simon.hentschel@uni-wuerzburg.de
#SBATCH --output=output/job-%j.out
#SBATCH --tmp=3000G


# #SBATCH --error=output/job-%j.err

export WANDB__SERVICE_WAIT=300
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
echo ${CUDA_VISIBLE_DEVICES}

echo "Copying HCLIM3 to scratch"
#measure runtime
start=$(date +%s.%N)
mkdir -p /tmp/Norway/HCLIM3/preprocessed_large/NorCP_AROME_EC-EARTH


#cp -n -r ~/datatemp/Norway/HCLIM3/preprocessed_large/NorCP_AROME_EC-EARTH /tmp/Norway/HCLIM3/preprocessed_large/NorCP_AROME_EC-EARTH
ssh alex "find /anvme/workspace/b214cb13-ecodata/downscaling/Norway/HCLIM3/preprocessed_large/NorCP_AROME_EC-EARTH -name '*.h5'" \
 | awk -F '[_.]' '{ year=$(NF-1); if (year >= 1980 && year <= 2005) print $0 }' \
 | xargs -P 1 -I{} rsync --info=PROGRESS2 alex:{} /tmp/Norway/HCLIM3/preprocessed_large/NorCP_AROME_EC-EARTH/.

cp ~/datatemp/Norway/HCLIM3/orog_NEU-3_ECMWF-ERAINT_evaluation_r1i1p1_HCLIMcom-HCLIM38-AROME_x2yn2v1_fx.nc /tmp/Norway/HCLIM3/.
cp ~/datatemp/Norway/HCLIM3/orog_NEU-3_ECMWF-ERAINT_evaluation_r1i1p1_HCLIMcom-HCLIM38-AROME_x2yn2v1_fx.nc /tmp/Norway/HCLIM3/preprocessed_large/.
cp ~/datatemp/Norway/HCLIM3/preprocessed_large/NorCP_AROME_EC-EARTH/* /tmp/Norway/HCLIM3/preprocessed_large/NorCP_AROME_EC-EARTH/.

wait


end=$(date +%s.%N)
echo "Copying data to scratch took $(awk "BEGIN {print ${end} - ${start}}") seconds"
echo "Copying data to scratch complete"



apptainer run --nv --bind /tmp/:/data --bind ~/bjerknes/corrdiff/checkpoints:/checkpoints/ ~/own_corrdiff.sif torchrun --nproc_per_node=8  ~/bjerknes/corrdiff/train.py --config-name $1 \

#apptainer run --nv --bind ~/data/downscaling:/data --bind ~/bjerknes/corrdiff/checkpoints:/checkpoints/ ~/own_corrdiff.sif torchrun --nproc_per_node=4  ~/bjerknes/corrdiff/train.py --config-name $1 \

