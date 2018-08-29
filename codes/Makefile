all: sbd

sbd: synthetic-benchmark-driver.cpp
	g++ synthetic-benchmark-driver.cpp -o sbd -lOpenCL -llsb -L$(LSB)/lib -I$(LSB)/include -std=c++11

test: sbd
	./sbd ./gpu-strided-global-memory-access-unoptimised.cl small 0 0 runtime;
	$(OCLGRIND)/bin/oclgrind --workload-characterisation ./sbd ./gpu-strided-global-memory-access-unoptimised.cl small 0 0 aiwc

clean:
	rm -f sbd lsb.*.r0* aiwc_*.csv aiwc_*_itb.log Rplots.pdf
