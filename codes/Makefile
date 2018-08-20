all: sbd

sbd: synthetic-benchmark-driver.cpp
	g++ synthetic-benchmark-driver.cpp -o sbd -lOpenCL -L$(LSB)/lib -I$(LSB)/include -std=c++11

clean:
	rm sbd
