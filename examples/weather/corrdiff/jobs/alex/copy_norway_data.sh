echo "Copying HCLIM3 to scratch"
#measure runtime
start=$(date +%s.%N)
mkdir -p /scratch/Norway/HCLIM3/preprocessed_large/


cp -n -r /anvme/workspace/b214cb13-ecodata/downscaling/Norway/HCLIM3/preprocessed_large/NorCP_AROME_EC-EARTH /scratch/Norway/HCLIM3/preprocessed_large/NorCP_AROME_EC-EARTH &
cp /anvme/workspace/b214cb13-ecodata/downscaling/Norway/HCLIM3/orog_NEU-3_ECMWF-ERAINT_evaluation_r1i1p1_HCLIMcom-HCLIM38-AROME_x2yn2v1_fx.nc /scratch/Norway/HCLIM3/.

wait


end=$(date +%s.%N)
echo "Copying data to scratch took $(awk "BEGIN {print ${end} - ${start}}") seconds"
echo "Copying data to scratch complete"

