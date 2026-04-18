#!/bin/bash


PYTHON=$1  
SCRIPTS=(
    "../preprocess/preprocess.py --datadir ../example --del_mode --preprocess"
    "../preprocess/esmif1_emb.py --datadir ../example/pdb"
    "../preprocess/integrate.py --datadir ../example"
)


for SCRIPT in "${SCRIPTS[@]}"; do
    $PYTHON $SCRIPT
done