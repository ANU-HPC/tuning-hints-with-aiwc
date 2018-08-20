
#Installation

This project uses Docker to facilitate reproducibility. As such, it has the following dependencies:

* Cuda 9.2 Runtime -- available [here](https://developer.nvidia.com/cuda-downloads)
* Docker -- available [here](https://docs.docker.com/install/linux/docker-ce/ubuntu/)
* nvidia-docker2, install instructions found [here](https://github.com/NVIDIA/nvidia-docker)
* Docker nvidia container, installed with: `sudo apt install nvidia-container-runtime`

#Build

To generate a docker image named guiding-optimisation, run:

`docker build -t guiding-optimisation .`

#Run

To start the docker image run:

`docker run --runtime=nvidia -it --mount src=`pwd`,target=/guiding-optimisation-with-aiwc,type=bind guiding-optimisation`

<!--
and run the demo with:
`bazel run //deeplearning/clgen -- --config /phd/deeplearning/clgen/tests/data/tiny/config.pbtxt`
--> 

