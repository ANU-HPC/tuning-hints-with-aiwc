# An Ubuntu environment configured for building the phd repo.
FROM nvidia/opencl:devel-ubuntu16.04

MAINTAINER Beau Johnston <beau.johnston@anu.edu.au>

# Disable post-install interactive configuration.
# For example, the package tzdata runs a post-installation prompt to select the
# timezone.
ENV DEBIAN_FRONTEND noninteractive

# use the closest Ubuntu mirror
RUN echo "deb mirror://mirrors.ubuntu.com/mirrors.txt bionic main restricted universe multiverse" > /etc/apt/sources.list \
 && echo "deb mirror://mirrors.ubuntu.com/mirrors.txt bionic-updates main restricted universe multiverse" >> /etc/apt/sources.list \
 && echo "deb mirror://mirrors.ubuntu.com/mirrors.txt bionic-security main restricted universe multiverse" >> /etc/apt/sources.list \
 && apt-get update

# Setup the environment.
ENV HOME /root
ENV USER docker

# Install essential packages.
RUN apt-get update && apt-get install --no-install-recommends -y \
    apt-utils \
    apt-transport-https \
    build-essential \
    cpio \
    file \
    git \
    less \
    make \
    ocl-icd-opencl-dev \
    pkg-config \
    sed \
    software-properties-common \
    wget \
    zlib1g-dev 

# Install cmake -- newer version than with apt
RUN wget -qO- "https://cmake.org/files/v3.12/cmake-3.12.1-Linux-x86_64.tar.gz" | tar --strip-components=1 -xz -C /usr

# Install Intel CPU Runtime for OpenCL Applications 18.1 for Linux (OpenCL 1.2)
RUN apt-get update && apt-get install -qqy \
    lsb-core \
    libnuma1 \
 && export RUNTIME_URL="http://registrationcenter-download.intel.com/akdlm/irc_nas/vcp/15532/l_opencl_p_18.1.0.015.tgz" \
    && export TAR=$(basename ${RUNTIME_URL}) \
    && export DIR=$(basename ${RUNTIME_URL} .tgz) \
    && wget -q ${RUNTIME_URL} \
    && tar -xf ${TAR} \
    && sed -i 's/decline/accept/g' ${DIR}/silent.cfg \
    && ${DIR}/install.sh --silent ${DIR}/silent.cfg \
;fi

# Install OpenCL Device Query tool
RUN git clone https://github.com/BeauJoh/opencl_device_query.git /opencl_device_query

# Install LibSciBench
ENV LSB /libscibench
ENV LSB_SRC /libscibench-source
RUN apt-get update && apt-get install --no-install-recommends -y \
    clang-5.0 \
    g++ \
    gcc \ 
    libclang-5.0-dev \
    llvm-5.0 \
    llvm-5.0-dev
RUN git clone https://github.com/spcl/liblsb.git $LSB_SRC
WORKDIR $LSB_SRC
RUN ./configure --prefix=$LSB
RUN make install

# Install leveldb (optional dependency for OclGrind)
ENV LEVELDB_SRC /leveldb-source
ENV LEVELDB_ROOT /leveldb
RUN git clone https://github.com/google/leveldb.git $LEVELDB_SRC
RUN mkdir $LEVELDB_SRC/build
WORKDIR $LEVELDB_SRC/build
RUN cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_INSTALL_PREFIX=$LEVELDB_ROOT
RUN make install

# Install OclGrind
ENV OCLGRIND_SRC /oclgrind-source
ENV OCLGRIND /oclgrind
ENV OCLGRIND_BIN /oclgrind/bin/oclgrind
RUN git clone https://github.com/ANU-HPC/Oclgrind.git $OCLGRIND_SRC
RUN mkdir $OCLGRIND_SRC/build
WORKDIR $OCLGRIND_SRC/build
ENV CC clang-5.0
ENV CXX clang++-5.0
RUN cmake $OCLGRIND_SRC -DUSE_LEVELDB=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo -DLLVM_DIR=/usr/lib/llvm5.0/lib/cmake -DCLANG_ROOT=/usr/lib/clang5.0 -DCMAKE_INSTALL_PREFIX=$OCLGRIND
RUN make install

# Install R and model dependencies
RUN apt-get update && apt-get install --no-install-recommends -y \
    dirmngr \
    gnupg-agent
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
RUN add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu bionic-cran35/'
RUN apt-get update && apt-get install --no-install-recommends -y \
    gfortran \
    libblas-dev \
    libcurl4-openssl-dev \
    liblapack-dev \
    libssl-dev \
    libxml2-dev \
    r-base
RUN Rscript -e "install.packages('devtools',repos = 'https://cloud.r-project.org');"
RUN Rscript -e "devtools::install_github('RcppCore/RcppEigen')"
RUN Rscript -e "devtools::install_github('imbs-hl/ranger')"

# Install the git-lsf module
ENV GIT_LSF /git-lsf
WORKDIR /downloads
RUN wget https://github.com/git-lfs/git-lfs/releases/download/v2.5.1/git-lfs-linux-amd64-v2.5.1.tar.gz
RUN mkdir $GIT_LSF
RUN tar -xvf git-lfs-linux-amd64-v2.5.1.tar.gz --directory $GIT_LSF
WORKDIR $GIT_LSF
RUN ./install.sh
RUN git lfs install

# Install the R model
ENV PREDICTIONS /opencl-predictions-with-aiwc
RUN git clone https://github.com/BeauJoh/opencl-predictions-with-aiwc.git $PREDICTIONS

# Install beakerx
RUN apt-get update && apt-get install --no-install-recommends -y \
    python3-dev \
    python3-pip \
    python3-setuptools \
    libbz2-dev \
    liblzma-dev \
    libpcre3-dev \
    libreadline-dev
RUN pip3 install --upgrade pip
RUN pip3 install tzlocal rpy2 pandas py4j ipywidgets requests beakerx \
    && beakerx install

# Install R module for beakerx
RUN Rscript -e "devtools::install_github('IRkernel/IRkernel')"
RUN Rscript -e "IRkernel::installspec(user = FALSE)"
RUN Rscript -e "devtools::install_github('tidyverse/magrittr')"
RUN Rscript -e "devtools::install_github('tidyverse/ggplot2')"
RUN Rscript -e "devtools::install_github('tidyverse/tidyr')"

# Install LetMeKnow
RUN pip3 install -U 'lmk==0.0.14'
# setup lmk by copying or add .lmkrc to /root/
# is used as: python3 ../opendwarf_grinder.py 2>&1 | lmk -
# or: lmk 'python3 ../opendwarf_grinder.py'

# Install EOD
ENV EOD /OpenDwarfs
RUN apt-get update && apt-get install --no-install-recommends -y \
    autoconf \
    automake \
    libtool
RUN git clone https://github.com/BeauJoh/OpenDwarfs.git $EOD
WORKDIR $EOD
RUN ./autogen.sh
RUN mkdir build
WORKDIR $EOD/build
RUN ../configure --with-libscibench=$LSB
RUN make

# Install Pandoc and Latex to build the paper
ENV PANDOC /pandoc
RUN apt-get update && apt-get install --no-install-recommends -y \
    lmodern \
    python-dev \
    python-pip \
    texlive-fonts-recommended \
    texlive-latex-extra \
    texlive-latex-recommended \
    texlive-science
RUN wget https://bootstrap.pypa.io/ez_setup.py -O - | python
RUN pip2 install setuptools && pip2 install wheel && pip2 install pandocfilters pandoc-fignos
WORKDIR $PANDOC
RUN wget https://github.com/jgm/pandoc/releases/download/1.19.2/pandoc-1.19.2-1-amd64.deb && apt-get install -y ./pandoc-1.19.2-1-amd64.deb
RUN wget https://github.com/lierdakil/pandoc-crossref/releases/download/v0.3.0.0/linux-ghc8-pandoc-2-0.tar.gz
RUN tar -xvf linux-ghc8-pandoc-2-0.tar.gz
RUN mv pandoc-crossref /usr/bin/

RUN apt-get update && apt-get install -y \
    curl \
    gdb \
    gdbserver \
    tree \
    vim

#container variables and startup...
WORKDIR /tuning-hints-with-aiwc
ENV LD_LIBRARY_PATH "${OCLGRIND}/lib:${LSB}/lib:./lib:${LD_LIBRARY_PATH}"
ENV PATH "${PATH}:${OCLGRIND}/bin"

RUN echo "export PATH=$PATH:$HOME/.cargo/bin" >> ~/.bashrc

#start beakerx/jupyter by default
#CMD ["beakerx","--allow-root"]

CMD ["/bin/bash"]
