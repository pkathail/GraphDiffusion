GraphDiffusion
--------------

GraphDiffusion is a python package based on the DiffusionGeometry library in Matlab (https://services.math.duke.edu/~mauro/code.html#DiffusionGeom). 

#### Installation and dependencies
1. GraphDiffusion has been implemented in Python3 and can be installed using

		$> git clone git@github.com:pkathail/GraphDiffusion.git
		$> cd GraphDiffusion
		$> sudo pip3 install

2. GraphDiffusion depends on a number of `python3` packages available on pypi and these dependencies are listed in `setup.py`
All the dependencies will be automatrically installed using the above commands

#### Usage
1. After installation, GraphDiffusion can be used with the following commands

		$> import GraphDiffusion
		$> res = GraphDiffusion.graph_diffusion.run_diffusion_map(...)
