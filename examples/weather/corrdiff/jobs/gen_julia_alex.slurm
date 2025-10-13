#!/bin/bash
#SBATCH -J PrecipitationDownscaling
#SBATCH -c 16
#SBATCH -p h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=simon.hentschel@uni-wuerzburg.de
#SBATCH --output=output/job-%j.out
#SBATCH --tmp=1000G


export WANDB__SERVICE_WAIT=300
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
echo ${CUDA_VISIBLE_DEVICES}


export MASTER_PORT=12345
export APPTAINERENV_HYDRA_FULL_ERROR=1

START_YEAR=2000
END_YEAR=2005

echo "Copying HCLIM3 to scratch"
#measure runtime
start=$(date +%s.%N)
mkdir -p /tmp/Norway/HCLIM3/preprocessed_large/NorCP_AROME_EC-EARTH


#cp -n -r ~/datatemp/Norway/HCLIM3/preprocessed_large/NorCP_AROME_EC-EARTH /tmp/Norway/HCLIM3/preprocessed_large/NorCP_AROME_EC-EARTH
ssh alex "find /anvme/workspace/b214cb13-ecodata/downscaling/Norway/HCLIM3/preprocessed_large/NorCP_AROME_EC-EARTH -name '*.h5'" \
 | awk -F '[_.]' '{ year=$(NF-1); if (year >= 2000 && year <= 2005) print $0 }' \
 | xargs -P 1 -I{} rsync --info=PROGRESS2 alex:{} /tmp/Norway/HCLIM3/preprocessed_large/NorCP_AROME_EC-EARTH/.

cp ~/datatemp/Norway/HCLIM3/orog_NEU-3_ECMWF-ERAINT_evaluation_r1i1p1_HCLIMcom-HCLIM38-AROME_x2yn2v1_fx.nc /tmp/Norway/HCLIM3/.
cp ~/datatemp/Norway/HCLIM3/orog_NEU-3_ECMWF-ERAINT_evaluation_r1i1p1_HCLIMcom-HCLIM38-AROME_x2yn2v1_fx.nc /tmp/Norway/HCLIM3/preprocessed_large/.
cp ~/datatemp/Norway/HCLIM3/preprocessed_large/NorCP_AROME_EC-EARTH/* /tmp/Norway/HCLIM3/preprocessed_large/NorCP_AROME_EC-EARTH/.

wait


end=$(date +%s.%N)
echo "Copying data to scratch took $(awk "BEGIN {print ${end} - ${start}}") seconds"
echo "Copying data to scratch complete"


srun apptainer run --nv --bind /tmp/:/data  \
                        --bind ~/bjerknes/corrdiff/checkpoints:/checkpoints/ \
                        ~/own_corrdiff.sif python -u ~/bjerknes/corrdiff/generate.py --config-name $1 \


