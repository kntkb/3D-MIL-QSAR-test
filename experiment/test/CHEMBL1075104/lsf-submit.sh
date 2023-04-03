#!/bin/bash
#BSUB -P "3dmil"
#BSUB -J "3dmil"
#BSUB -n 1
#BSUB -R rusage[mem=64]
#BSUB -R span[hosts=1]
###BSUB -q cpuqueue
#BSUB -q gpuqueue
#BSUB -sp 1 # low priority. default is 12, max is 25
#BSUB -gpu num=1:j_exclusive=yes:mode=shared
#BSUB -W 3:00
#BSUB -L /bin/bash
#BSUB -o out_%J_%I.stdout
#BSUB -eo out_%J_%I.stderr

source ~/.bashrc
#OPENMM_CPU_THREADS=1
#export OE_LICENSE=~/.openeye/oe_license.txt   # Open eye license activation/env

# change dir
echo "changing directory to ${LS_SUBCWD}"
cd $LS_SUBCWD


# Report node in use
echo "======================"
hostname
env | sort | grep 'CUDA'
nvidia-smi
echo "======================"


conda activate 3dmil
script_path="/home/takabak/data/3d-mil-qsar-test"

descriptor_method="pmapper"
dataset="CHEMBL1075104"
path_to_dataset="../../../datasets"
wrapper_method="instance"
python ${script_path}/3dmil.py --path_to_dataset ${path_to_dataset} --dataset ${dataset} --descriptor_method ${descriptor_method} --wrapper_method ${wrapper_method}