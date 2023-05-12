#!/bin/bash

DIR=${PWD}
ls -d classA_lipid* > list

while read line;
do
    echo "${line}"
    cd ${line}
    #echo "bsub < lsf-submit.sh"
    bsub < lsf-submit.sh
    cd ${DIR}
done < list

rm list