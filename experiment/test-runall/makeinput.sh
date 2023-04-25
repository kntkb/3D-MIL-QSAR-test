#!/bin/bash

DATASET="CHEMBL1075104"

mkdir -p ${DATASET}
ml_models=('AttentionNetRegressor' 'BagWrapperMLPRegressor' 'InstanceWrapperMLPRegressor' 'RandomForestRegressor-Morgan' 'RandomForestRegressor-RDKit2D')
for ml_model in ${ml_models[*]};
do
    echo ">${ml_model}"
    mkdir ${DATASET}/${ml_model}
    sed -e 's/@@@DATASET@@@/'${DATASET}'/' \
        templates/config_${ml_model}.yml > ${DATASET}/${ml_model}/config.yml
done

cp templates/lsf-submit.sh ${DATASET}/lsf-submit.sh