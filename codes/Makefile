all: gpu-stride cpu-loop-block cpu-mandelbrot-vectorization aiwc-tester #fdtd 

gpu-stride: gpu-strided-global-memory-access-unoptimised.cpp
	g++ gpu-strided-global-memory-access-unoptimised.cpp -o gpu-stride -lOpenCL -llsb -L$(LSB)/lib -I$(LSB)/include -std=c++11

cpu-loop-block: cpu-loop-blocking.cpp
	g++ cpu-loop-blocking.cpp -o cpu-loop-block -lOpenCL -llsb -L$(LSB)/lib -I$(LSB)/include -std=c++11

cpu-mandelbrot-vectorization: cpu-mandelbrot-vectorization.cpp
	g++ cpu-mandelbrot-vectorization.cpp -o cpu-mandelbrot-vectorization -lOpenCL -llsb -L$(LSB)/lib -I$(LSB)/include -std=c++11 -g -O0

fdtd: $(LIB) oclFDTD3d/src/oclFDTD3d.cpp oclFDTD3d/src/FDTD3dReference.cpp oclFDTD3d/src/FDTD3dGPU.cpp $(LIB)/libshrUtils.so $(LIB)/liboclUtils.so
	g++ -IoclFDTD3d/inc -IoclFDTD3d/src oclFDTD3d/src/oclFDTD3d.cpp oclFDTD3d/src/FDTD3dReference.cpp oclFDTD3d/src/FDTD3dGPU.cpp -o fdtd -lOpenCL -llsb -loclUtils -lshrUtils -Llib -L$(LSB)/lib -I$(LSB)/include -std=c++11

$(LIB)/libshrUtils.so: oclFDTD3d/src/shrUtils.cpp oclFDTD3d/src/cmd_arg_reader.cpp
	g++ -shared -fPIC -IoclFDTD3d/inc -IoclFDTD3d/src oclFDTD3d/src/shrUtils.cpp oclFDTD3d/src/cmd_arg_reader.cpp -o $(LIB)/libshrUTils.so -lOpenCL -std=c++11 -DUNIX

$(LIB)/liboclUtils.so:  oclFDTD3d/src/oclUtils.cpp
	g++ -shared -fPIC -IoclFDTD3d/inc -IoclFDTD3d/src oclFDTD3d/src/oclUtils.cpp -o $(LIB)/liboclUtils.so -lOpenCL -std=c++11 -DUNIX

$(LIB):
	mkdir -p lib

test: gpu-stride
	./gpu-stride ./gpu-strided-global-memory-access-unoptimised.cl small 0 0 runtime;
	$(OCLGRIND)/bin/oclgrind --workload-characterisation ./gpu-stride ./gpu-strided-global-memory-access-unoptimised.cl small 0 0 aiwc

aiwc-tester: aiwc-tester.cpp
	g++ aiwc-tester.cpp -o aiwc-tester -lOpenCL -llsb -L$(LSB)/lib -I$(LSB)/include -std=c++11

aiwc-test: aiwc-tester
	$(OCLGRIND)/bin/oclgrind --workload-characterisation ./aiwc-tester ./simple-aiwc-tests.cl tiny 0 0 aiwc

clean:
	rm -f sbd aiwc-tester lsb.*.r0* aiwc_*.csv aiwc_*_itb.log Rplots.pdf cpu-loop-block cpu-mandelbrot-vectorization
