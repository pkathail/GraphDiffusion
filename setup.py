from setuptools import setup


setup(name='GraphDiffusion',
      version='1.0',
      description='python package based on the DiffusionGeometry library in Matlab (https://services.math.duke.edu/~mauro/code.html#DiffusionGeom)',
      author='Pooja Kathail',
      author_email='pk2485@columbia.edu',
      package_dir={'': 'src'},
      install_requires=[
          'numpy>=1.10.0',
          'pandas>=0.18.0',
          'scipy>=0.14.0',
          'sklearn']
      )
