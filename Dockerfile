# An Ubuntu environment configured for building the phd repo.
FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04
#FROM ubuntu:16.04

MAINTAINER Beau Johnston <beau.johnston@anu.edu.au>

# Disable post-install interactive configuration.
# For example, the package tzdata runs a post-installation prompt to select the
# timezone.
ENV DEBIAN_FRONTEND noninteractive

# Setup the environment.
ENV HOME /root
ENV USER docker
ENV LSB_SRC /libscibench-source
ENV LSB /libscibench
ENV OCLGRIND_SRC /oclgrind-source
ENV OCLGRIND /oclgrind
ENV PREDICTIONS /opencl-predictions-with-aiwc

# Install essential packages.
RUN apt-get update
RUN apt-get install --no-install-recommends -y software-properties-common ocl-icd-opencl-dev

# Install LibSciBench
RUN apt-get install --no-install-recommends -y git cmake llvm-3.9 llvm-3.9-dev clang-3.9 libclang-3.9-dev gcc g++ make zlib1g-dev
RUN git clone https://github.com/spcl/liblsb.git $LSB_SRC

WORKDIR $LSB_SRC
RUN ./configure --prefix=$LSB
RUN make
RUN make install

# Install OclGrind
RUN git clone https://github.com/BeauJoh/Oclgrind.git $OCLGRIND_SRC

RUN mkdir $OCLGRIND_SRC/build
WORKDIR $OCLGRIND_SRC/build
ENV CC clang-3.9
ENV CXX clang++-3.9

RUN cmake $OCLGRIND_SRC -DCMAKE_BUILD_TYPE=RelWithDebInfo -DLLVM_DIR=/usr/lib/llvm-3.9/lib/cmake -DCLANG_ROOT=/usr/lib/clang/3.9.1 -DCMAKE_INSTALL_PREFIX=$OCLGRIND

RUN make
RUN make install

# Install R and model dependencies
RUN apt-get install --no-install-recommends -y r-base libcurl4-openssl-dev libssl-dev r-cran-rcppeigen
RUN Rscript -e "install.packages('devtools',repos = 'http://cran.us.r-project.org');"
RUN Rscript -e "devtools::install_github('imbs-hl/ranger')"
RUN git clone https://github.com/BeauJoh/opencl-predictions-with-aiwc.git $PREDICTIONS

CMD ["/bin/bash"]
