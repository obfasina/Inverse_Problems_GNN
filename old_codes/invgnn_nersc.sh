#!/bin/bash

conda env create -f environment.yml
conda activate inv_gnnenv
python train_FEM.py
