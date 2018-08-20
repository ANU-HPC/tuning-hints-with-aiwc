
#include <iostream>
#include <cstring>
#include <cassert>
#include <fstream>
#include <ctime>
#include <cstdlib>
#include <random>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#ifdef __APPLE__
    #include <OpenCL/cl.h>
#else
    #include <CL/opencl.h>
#endif

const float EPSILON = 0.00001f;

inline void except(bool condition, const std::string &error_message = "")
{
    if (!condition)
        throw std::runtime_error(error_message);
}

inline void zero_payload(float*x, unsigned int size){
    for(int i = 0; i < size; i++){
        x[i] = 0.0f;
    }
}

inline void randomise_payload(float*x,unsigned int size){
    std::random_device seed;
    std::mt19937 gen(seed());
    std::uniform_int_distribution<int> dist(0, 100);

    for(int i = 0; i < size; i++){
        x[i] = dist(gen);
    }
}

inline void copy_payload(float*in,float*out,unsigned int size){
    for(int i = 0; i < size; i++){
        out[i] = in[i];
    }
}

bool same_payload(float* in, float* out, unsigned int size){
    for(int i = 0; i < size; i++){
        if (abs(out[i] - in[i]) > EPSILON){
            return false;
        }
    }
    return true;
}

bool different_payload(float*in, float* out, unsigned int size){
    return(!(same_payload(in,out,size)));
}

inline void print_payload(float*x,unsigned int size){
    for(int i = 0; i < size; i++){
        std::cout << x[i] << ' ';
    }
    std::cout << std::endl;
}

int main(int argc, char** argv){

    //extract kernel -- ./sbd <kernel source> <tiny/small/medium/large> <platform id> <device id>
    except(argc == 5, "./sbd <kernel source> <tiny/small/medium/large> <platform id> <device id>");
    char* synthetic_kernel_path = argv[1];
    char* problem_size = argv[2];
    int platform_id = atoi(argv[3]);
    int device_id = atoi(argv[4]);

    // read synthetic kernel file
    std::ifstream sk_handle(synthetic_kernel_path, std::ios::in);
    except(sk_handle.is_open(), "synthetic kernel doesn\'t exist");
    std::cout << "Attempting kernel: " << synthetic_kernel_path << " with contents:\n" << sk_handle.rdbuf() << std::endl;
    std::filebuf* sk_buf = sk_handle.rdbuf();
    int sk_size = sk_buf->pubseekoff(0,sk_handle.end,sk_handle.in);
    sk_buf->pubseekpos(0,sk_handle.in);
    char* sk_source = new char[sk_size];
    sk_buf->sgetn(sk_source,sk_size);

    //set-up open compute language
    int sbd_err;
    cl_uint num_platforms = 0;
    cl_uint num_devices = 0;

    cl_device_id* sbd_devices;
    cl_platform_id* sbd_platforms;

    cl_context sbd_context;
    cl_command_queue sbd_queue;
   
    sbd_err = clGetPlatformIDs(0, NULL, &num_platforms);
    except(sbd_err == CL_SUCCESS, "can't get platform counts");
    sbd_platforms = new cl_platform_id[num_platforms];
    sbd_err = clGetPlatformIDs(num_platforms, sbd_platforms, NULL);
    except(sbd_err == CL_SUCCESS, "can't get platform info");
    except(num_platforms, "no OpenCL platforms found");
    std::cout << platform_id << std::endl;
    std::cout << num_platforms << std::endl;
    except(platform_id >= 0 && platform_id < num_platforms, "invalid platform selection");

    sbd_err = clGetDeviceIDs(sbd_platforms[platform_id], CL_DEVICE_TYPE_ALL,0, 0, &num_devices);
    except(sbd_err == CL_SUCCESS, "can't get device counts");
    except(num_devices, "no OpenCL devices found");
    sbd_devices = new cl_device_id[num_devices];
    sbd_err = clGetDeviceIDs(sbd_platforms[platform_id], CL_DEVICE_TYPE_ALL, num_devices, sbd_devices, NULL);
    except(sbd_err == CL_SUCCESS, "can't get device info");
    except(device_id >= 0 && device_id < num_devices, "invalid device selection");

    sbd_context = clCreateContext(0, 1, &sbd_devices[device_id], NULL, NULL, &sbd_err);
    except(sbd_err == CL_SUCCESS, "can't create context");
    sbd_queue = clCreateCommandQueue(sbd_context, sbd_devices[device_id], 0, &sbd_err);
    except(sbd_err == CL_SUCCESS, "can't create command queue");

    //compile synthetic kernel
    cl_program sbd_program = clCreateProgramWithSource(sbd_context, 1, (const char **) &sk_source, NULL, &sbd_err);
    except(sbd_err == CL_SUCCESS, "can't build kernel");
    sbd_err = clBuildProgram(sbd_program, 1, &sbd_devices[device_id], NULL, NULL, NULL);
    except(sbd_err == CL_SUCCESS, "can't build program");
    cl_kernel sbd_kernel = clCreateKernel(sbd_program, "A", &sbd_err);
    except(sbd_err == CL_SUCCESS, "can't create memset kernel");

    //set-up memory for payload/problem size
    size_t KiB;
    if(strcmp(problem_size, "tiny"))       {KiB = 31;}    //  32 KiB < L1
    else if(strcmp(problem_size, "small")) {KiB = 255;}   // 256 KiB < L2
    else if(strcmp(problem_size, "medium")){KiB = 8191;}  //8192 KiB < L3
    else if(strcmp(problem_size, "large")) {KiB = 16384;} //8192 KiB > L3 
    else{assert(false && "invalid problem size -- must be tiny, small, medium or large");} 

    float bytes_per_buffer = (KiB*1024)/3;
    cl_int elements = static_cast<cl_int>(bytes_per_buffer/sizeof(float));
    cl_mem sbd_a = clCreateBuffer(sbd_context,CL_MEM_READ_WRITE,bytes_per_buffer,NULL,&sbd_err);
    except(sbd_err == CL_SUCCESS, "can't create device memory a");
    cl_mem sbd_b = clCreateBuffer(sbd_context,CL_MEM_READ_WRITE,bytes_per_buffer,NULL,&sbd_err);
    except(sbd_err == CL_SUCCESS, "can't create device memory b");
    cl_mem sbd_c = clCreateBuffer(sbd_context,CL_MEM_READ_WRITE,bytes_per_buffer,NULL,&sbd_err);
    except(sbd_err == CL_SUCCESS, "can't create device memory c");

    //dynamic checker
    unsigned int total_payload = elements*3;//kernel arguments are set using 3 even length argument buffers at once
    float* x1_in  = new float[total_payload]; 
    float* x1_out = new float[total_payload];
    float* x2_in  = new float[total_payload];
    float* x2_out = new float[total_payload];
    float* y1_in  = new float[total_payload];
    float* y1_out = new float[total_payload];
    float* y2_in  = new float[total_payload];
    float* y2_out = new float[total_payload];

    randomise_payload(x1_in,total_payload);
    randomise_payload(y1_in,total_payload);
    copy_payload(x1_in,x2_in,total_payload);
    copy_payload(y1_in,y2_in,total_payload);

    //confirm payload generation conditions
    assert(same_payload(x1_in,x2_in,total_payload));
    assert(same_payload(y1_in,y2_in,total_payload));
    assert(different_payload(x1_in,y1_in,total_payload));

    //run the kernel 4 times to assess if it is performing useful work
    size_t global_work = elements;
    size_t local_work = 1; 

    for(int i = 0; i <= 3; i++){
        float* in; float* out;
        switch(i){
            case 0: in = x1_in; out = x1_out; break;
            case 1: in = y1_in; out = y1_out; break;
            case 2: in = x2_in; out = x2_out; break;
            case 3: in = y2_in; out = y2_out; break;
        }
        sbd_err  = clEnqueueWriteBuffer(sbd_queue,sbd_a,CL_TRUE,0,bytes_per_buffer,in,0,NULL,NULL);
        sbd_err |= clEnqueueWriteBuffer(sbd_queue,sbd_b,CL_TRUE,0,bytes_per_buffer,in+elements,0,NULL,NULL);
        sbd_err |= clEnqueueWriteBuffer(sbd_queue,sbd_c,CL_TRUE,0,bytes_per_buffer,in+2*elements,0,NULL,NULL);
        except(sbd_err == CL_SUCCESS, "can't write to device memory!");

        sbd_err = clSetKernelArg(sbd_kernel, 0, sizeof(cl_mem), &sbd_a);
        sbd_err = clSetKernelArg(sbd_kernel, 1, sizeof(cl_mem), &sbd_b);
        sbd_err = clSetKernelArg(sbd_kernel, 2, sizeof(cl_mem), &sbd_c);
        sbd_err = clSetKernelArg(sbd_kernel, 3, sizeof(cl_int), &elements);
        except(sbd_err == CL_SUCCESS, "failed to set kernel arguments");

        sbd_err = clEnqueueNDRangeKernel(sbd_queue, sbd_kernel, 1, NULL, &global_work,&local_work,0,NULL,NULL);
        except(sbd_err == CL_SUCCESS, "failed to execute kernel");

        clFinish(sbd_queue);

        sbd_err  = clEnqueueReadBuffer(sbd_queue,sbd_a,CL_TRUE,0,bytes_per_buffer,out,0,NULL,NULL);
        sbd_err |= clEnqueueReadBuffer(sbd_queue,sbd_b,CL_TRUE,0,bytes_per_buffer,out+elements,0,NULL,NULL);
        sbd_err |= clEnqueueReadBuffer(sbd_queue,sbd_c,CL_TRUE,0,bytes_per_buffer,out+2*elements,0,NULL,NULL);
        except(sbd_err == CL_SUCCESS, "can't read from device memory");
    }

    //assert kernel has valid output for at least one of these vectors
    assert(different_payload(x1_out,x1_in,total_payload) && different_payload(y1_out,y1_in,total_payload));
    print_payload(x1_out,10); std::cout << std::endl;
    print_payload(y1_out,10); std::cout << std::endl;
    //assert kernel is input sensitive
    assert(different_payload(x1_out,y1_out,total_payload) && different_payload(x2_out,y2_out,total_payload));
    //assert kernel is deterministic
    assert(same_payload(x1_out,x2_out,total_payload) && same_payload(y1_out,y2_out,total_payload));

    delete x1_in;
    delete x1_out;
    delete y1_in;
    delete y1_out;
    delete x2_in;
    delete x2_out;
    delete y2_in;
    delete y2_out;

    clReleaseMemObject(sbd_c);
    clReleaseMemObject(sbd_b);
    clReleaseMemObject(sbd_a);
    clReleaseKernel(sbd_kernel);
    clReleaseProgram(sbd_program);
    clReleaseCommandQueue(sbd_queue);
    clReleaseContext(sbd_context);
    delete sk_source;
    delete sbd_devices;
    delete sbd_platforms;   
}

