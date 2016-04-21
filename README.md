GraphDiffusion
--------------

GraphDiffusion is a python package based on the DiffusionGeometry library in Matlab (https://services.math.duke.edu/~mauro/code.html#DiffusionGeom). 

#### Installation and dependencies
1. GraphDiffusion has been implemented in Python3 and can be installed using

		$> git clone git@github.com:pkathail/GraphDiffusion.git
		$> cd GraphDiffusion
		$> sudo pip3 install .

2. GraphDiffusion depends on a number of `python3` packages available on pypi and these dependencies are listed in `setup.py`
All the dependencies will be automatrically installed using the above commands

#### Usage
After installation, GraphDiffusion can be used with the following commands

		$> import GraphDiffusion
		$> res = GraphDiffusion.graph_diffusion.run_diffusion_map(data, knn=10, normalization='smarkov')

where `data` is a `N x D` matrix representing `N` points in `R ^ D`, `knn` is the number of nearest neighbors, `normalization` is the method for normalizing weights. Please refer to the docstring for more details. 

`res` is dictionary with the following objects
1. `T`: `N x N` sparse matrix giving the normalized diffusion operator
2. `W`: `N x N` sparse matrix of weights
3. `EigenVectors`: Eigen vectors of matrix `T`
4. `EigenValues`: Eigen values of matrix `T`