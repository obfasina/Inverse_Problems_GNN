#!/bin/sh

cd ~/Documents/sfepy-master

for i in {2999..3000}; do
	sfepy-run simple /Users/oluwadamilolafasina/Documents/sfepy-master/sfepy/examples/diffusion/poisson_neumann_edit.py -d prm=$i
	export i
	python -c "import meshio; sol = meshio.read('/Users/oluwadamilolafasina/Documents/sfepy-master/cross-51-0.34.vtk'); import numpy as np; import os; np.save('poisneumann'+ 
str(os.environ['i']) 
,sol.point_data['t'])"
done
