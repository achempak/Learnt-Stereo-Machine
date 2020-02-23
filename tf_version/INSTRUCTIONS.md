#OLD
1. Clone https://github.com/akar43/lsm.git somewhere.
2. Copy docker.sh and Dockerfile into this folder. Also, overwrite the existing download_sample.sh and get_models.sh with the corresponding files in this repo.
3. Install nvidia-docker with this link: https://github.com/NVIDIA/nvidia-docker. This will require installation of a CUDA driver.
4. Use the Dockerfile to create an image with the correct environment for running this project.
5. Download https://drive.google.com/open?id=1qg0t_nKMBMz_GhyIUdBK2l_aNqfFewNe. Modify download_sample.sh to include the path where you downloaded the file to. Temporarily comment out the "exit 0" line and run the script. Uncomment this line after running.
6. Download https://drive.google.com/file/d/1HAZuKWYxYybDMqb6RA5PLvUes52x3dhm/view?usp=sharing. Modify get_models.sh to include the path where you downloaded the file to. Temporarily comment out the "exit 0" line and run the script. Uncomment this line after running.
7. Modify the line in docker.sh from '-v "$(pwd)":/home/adityac/lsm' to wherever you want to mount the project file in your Docker container. Change 'nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04' to whatever you named your Docker image.
8. cd to the lsm folder and run 'jupyter notebook --ip 0.0.0.0 --no-browser --allow-root'.

#NEW
1. Clone https://github.com/akar43/lsm.git somewhere.
2. Create anaconda environment using python 2.7 with the libraries in requirements.txt. Install tensorflow-gpu=1.15 and cudatoolkit-10.0.
5. Download https://drive.google.com/open?id=1qg0t_nKMBMz_GhyIUdBK2l_aNqfFewNe. Modify download_sample.sh to include the path where you downloaded the file to. Temporarily comment out the "exit 0" line and run the script. Uncomment this line after running.
6. Download https://drive.google.com/file/d/1HAZuKWYxYybDMqb6RA5PLvUes52x3dhm/view?usp=sharing. Modify get_models.sh to include the path where you downloaded the file to. Temporarily comment out the "exit 0" line and run the script. Uncomment this line after running.
7. Download https://drive.google.com/open?id=17GjULuQZsn-s92PQFQSBzezDkonowIxR. Modify prepare_data.sh in the same manner as above.
8. Follow the README on the lsm github repo to train, test, and validate the NN.
