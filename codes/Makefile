all: sbc sbd

sbd: synthetic-benchmark-driver.cpp
	g++ synthetic-benchmark-driver.cpp -o sbd -lOpenCL -L$(LSB)/lib -I$(LSB)/include -std=c++11

sbc: synthetic-benchmark-checker.cpp
	g++ synthetic-benchmark-checker.cpp -o sbc -lOpenCL -std=c++11

clean:
	rm sbd sbc
