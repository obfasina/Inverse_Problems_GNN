# Inverse_Problems_GNN

Three main components of inverse solver:
a) Training script: "train_FEMgpuGS.py"
b) FEM forward solver: "fenics_datagen.py"
c) Script that applies Geometric Scattering Transform: "apply_GS.py"

Steps to run: 

1) Create environment using:  "conda create --name myenv --file spec_file.txt"
2) Activate environment using: "conda activate myenv"

All packages/depencies should be available in the specified envrionment following those two commands.

Running inverse solver:

On the CLI, make sure you are in the Inverse_GNN directrory, then run:

python train_GEMgpuGS.py 

Four main flags in CLI:

a) --FEM_solver: Will incorporate FEM loss into trainings  
b) --wandb: Specify whether you want wandb to track results
c) --GS: inclusion of flag applies geometric scattering transfrom
d) --new_run_data: inclusion of flag generates new training data 
