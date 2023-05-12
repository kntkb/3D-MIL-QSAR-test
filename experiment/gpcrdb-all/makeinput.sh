#!/bin/bash


# STEP1: CONVERT ALL xxxx.smi INTO ARRAY
declare -a array=()   # declare an empty array
while read line;
do
    #echo "$line"
    x=$(echo "${line%.smi}")
    #echo "$x"
    array+=("$x")     # append x to the end of the array
done < datasetlist
#echo "Array contents: ${array[@]}"



# STEP2: LOOP OVER DATASETS AND CREATE INPUTS FOR EACH ML METHOD
ml_models=('AttentionNetRegressor' 'BagNetRegressor' 'BagNetRegressor-lse' 'InstanceNetRegressor' 'InstanceNetRegressor-lse' 'BagWrapperMLPRegressor' 'InstanceWrapperMLPRegressor' 'RandomForestRegressor-Morgan' 'RandomForestRegressor-RDKit2D')
for DATASET in "${array[@]}"
do
    echo "$DATASET"
    mkdir -p ${DATASET}
    for ml_model in ${ml_models[*]};
    do
        echo ">${ml_model}"
        mkdir ${DATASET}/${ml_model}
        sed -e 's/@@@DATASET@@@/'${DATASET}'/' \
            templates/config_${ml_model}.yml > ${DATASET}/${ml_model}/config.yml
    done
    cp templates/lsf-submit.sh ${DATASET}/lsf-submit.sh
done

