# An Ubuntu environment configured for building the phd repo.
FROM nvidia/opencl
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
RUN apt-get install --no-install-recommends -y software-properties-common \
    ocl-icd-opencl-dev \
    pkg-config \
    build-essential \
    git \
    cmake \
    make \
    zlib1g-dev \
    apt-transport-https

# Install OpenCL Device Query tool
RUN git clone https://github.com/BeauJoh/opencl_device_query.git /opencl_device_query

# Install LibSciBench
RUN apt-get install --no-install-recommends -y llvm-3.9 llvm-3.9-dev clang-3.9 libclang-3.9-dev gcc g++
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
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
RUN add-apt-repository 'deb [arch=amd64,i386] https://cran.rstudio.com/bin/linux/ubuntu xenial/'
RUN apt-get update
RUN apt-get install --no-install-recommends -y r-base libcurl4-openssl-dev libssl-dev r-cran-rcppeigen
RUN Rscript -e "install.packages('devtools',repos = 'http://cran.us.r-project.org');"
RUN Rscript -e "devtools::install_github('imbs-hl/ranger')"
RUN git clone https://github.com/BeauJoh/opencl-predictions-with-aiwc.git $PREDICTIONS

# Install beakerx
RUN apt-get install --no-install-recommends -y python3-pip python3-setuptools python3-dev libreadline-dev libpcre3-dev libbz2-dev liblzma-dev
RUN pip3 install --upgrade pip
RUN pip3 install tzlocal rpy2 requests beakerx \
    && beakerx install

# Install R module for beakerx
RUN Rscript -e "devtools::install_github('IRkernel/IRkernel')"
RUN Rscript -e "IRkernel::installspec(user = FALSE)"
RUN Rscript -e "devtools::install_github('tidyverse/magrittr')"
RUN Rscript -e "devtools::install_github('tidyverse/ggplot2')"
RUN Rscript -e "devtools::install_github('tidyverse/tidyr')"

CMD ["/bin/bash"]

WORKDIR /guiding-optimisation-with-aiwc
ENV LD_LIBRARY_PATH "${OCLGRIND}/lib:${LSB}/lib:${LD_LIBRARYPATH}"
ENV PATH "${PATH}:${OCLGRIND}/bin}"

