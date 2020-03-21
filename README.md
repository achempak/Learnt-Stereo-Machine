# Learnt-Stereo-Machine

Because we live in a 3-dimensional world, it is generally significantly easier to work with a 3D representations of an object (e.g. point cloud, voxel grid, etc) than through corresponding 2D representations (e.g. photographs). Ideally, what we desire is some way to transform these 2D representations into a 3D representation. Classical methods of doing this typically reduce to finding dense correspondences in images (of the same object) and triangulating to find a 3D reconstruction. 

Kar et al. tackle the 3D reconstruction problem from a different angle: machine learning \cite{lsm}. The authors propose and implement a neural network that can take as input one or more images of an object and produce a 3D voxel representation of that object. In this project, we re-implement this neural network in PyTorch as well as attempt to improve it with minor modifications. While the authors only test their network on the ShapeNet dataset \cite{shapenet2015}, we also test its efficacy on a second dataset, ModelNet10.

# Installation

Simply clone this repository and run `easy_install.sh`.

# Running

To run a pre-trained model, simply run the Demo.ipynb notebook in the lsm directory.
