
#include <iostream>
#include <cstring>
#include <cassert>
#include <fstream>
#include <ctime>
#include <cstdlib>
#include <random>
#include <liblsb.h>

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
    except(argc == 6, "./sbd <kernel source> <tiny/small/medium/large> <platform id> <device id> <aiwc/runtime>");
    char* synthetic_kernel_path = argv[1];
    char* problem_size = argv[2];
    int platform_id = atoi(argv[3]);
    int device_id = atoi(argv[4]);
    char* mode = argv[5];

    LSB_Init("cpu_loop_blocking",0);
    LSB_Set_Rparam_string("problem_size",problem_size);
    LSB_Set_Rparam_string("kernel","none_yet");

    LSB_Set_Rparam_string("region", "host_side_setup");
    LSB_Res();
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

    //set-up memory for payload/problem size
    size_t KiB;
    if(strcmp(problem_size, "tiny")==0)       {KiB = 31;}    //  32 KiB < L1
    else if(strcmp(problem_size, "small")==0) {KiB = 255;}   // 256 KiB < L2
    else if(strcmp(problem_size, "medium")==0){KiB = 7900;}  //8192 KiB < L3
    else if(strcmp(problem_size, "large")==0) {KiB = 16384;} //8192 KiB > L3 
    else if(strcmp(problem_size, "huge")==0)  {KiB = 131072;}//
    else{assert(false && "invalid problem size -- must be tiny, small, medium or large");} 

    unsigned int c_bytes = (KiB*1024);
    cl_int c_elements = static_cast<cl_int>(c_bytes/sizeof(float));
    //MxN matrix (but actually square matrix)
    int w = 32;
    int M = floor(sqrt(c_elements));
    M = floor(M/w)*w; //but rounded down so it's a multiple of 32 -- 32x32 divisible blocks

    unsigned int a_bytes = M*M*sizeof(float);
    unsigned int b_bytes = M*M*sizeof(float);
    w = 1;

    std::cout << "M = " << M << " total KiB = " <<  (a_bytes+b_bytes)/1024 << std::endl;

    LSB_Rec(0);

    

    std::cout << "Operating on a " << M << "x" << M << " matrix with a tile size " << w << "..." << std::endl;

    LSB_Set_Rparam_string("region", "kernel_creation");
    LSB_Res();
    //compile kernels
    std::string compiler_flags = "-DMAX=" + std::to_string(M) + " -DBLOCK_SIZE=8"; 
    cl_program sbd_program = clCreateProgramWithSource(sbd_context, 1, (const char **) &sk_source, NULL, &sbd_err);
    except(sbd_err == CL_SUCCESS, "can't build kernel");
    sbd_err = clBuildProgram(sbd_program, 1, &sbd_devices[device_id], compiler_flags.c_str(), NULL, NULL);
    if(sbd_err != CL_SUCCESS){//print error during kernel compilation
        size_t log_size;
        clGetProgramBuildInfo(sbd_program, sbd_devices[device_id], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* build_log = new char[log_size];  
        clGetProgramBuildInfo(sbd_program, sbd_devices[device_id], CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
        std::cerr << "Error in kernel: " << std::endl;
        std::cerr << build_log << std::endl;
        delete[] build_log;
    }
    except(sbd_err == CL_SUCCESS, "can't build program");
    cl_kernel originalLoop_kernel = clCreateKernel(sbd_program, "original_loop", &sbd_err);
    except(sbd_err == CL_SUCCESS, "can't create kernel");
    cl_kernel transformedLoop_kernel = clCreateKernel(sbd_program, "transformed_loop_after_blocking", &sbd_err);
    except(sbd_err == CL_SUCCESS, "can't create kernel");
    LSB_Rec(0);

    //memory setup
    LSB_Set_Rparam_string("region", "device_side_buffer_setup");
    LSB_Res();
    cl_mem sbd_a = clCreateBuffer(sbd_context,CL_MEM_READ_WRITE,a_bytes,NULL,&sbd_err);
    except(sbd_err == CL_SUCCESS, "can't create device memory a");
    cl_mem sbd_b = clCreateBuffer(sbd_context,CL_MEM_READ_WRITE,b_bytes,NULL,&sbd_err);
    except(sbd_err == CL_SUCCESS, "can't create device memory b");

    float* a  = new float[M*M]; 
    float* b  = new float[M*M];
    LSB_Rec(0);

    int sample_size = 100;
    if(strcmp(mode, "aiwc")==0){
        sample_size = 1;
    }
    //simple case
    LSB_Set_Rparam_string("kernel","original_loop");
    for(int i = 0; i < sample_size; i++){
        LSB_Set_Rparam_string("region", "host_side_initialise_buffers");
        randomise_payload(a,M*M);
        randomise_payload(b,M*M);
        LSB_Rec(i);

        LSB_Set_Rparam_string("region","device_side_h2d_copy");
        LSB_Res();
        sbd_err  = clEnqueueWriteBuffer(sbd_queue,sbd_a,CL_TRUE,0,a_bytes,a,0,NULL,NULL);
        sbd_err |= clEnqueueWriteBuffer(sbd_queue,sbd_b,CL_TRUE,0,b_bytes,b,0,NULL,NULL);
        except(sbd_err == CL_SUCCESS, "can't write to device memory!");
        LSB_Rec(i);

        //run the kernel
        size_t global_work[2] = {static_cast<size_t>(M),static_cast<size_t>(M)};
        //size_t local_work[2] = {static_cast<size_t>(w),static_cast<size_t>(w)}; 
        //on one thread
        size_t local_work[2] = {static_cast<size_t>(M),static_cast<size_t>(M)}; 

        LSB_Set_Rparam_string("region","original_loop_kernel");
        LSB_Res();
        sbd_err  = clSetKernelArg(originalLoop_kernel, 0, sizeof(cl_mem), &sbd_a);
        sbd_err |= clSetKernelArg(originalLoop_kernel, 1, sizeof(cl_mem), &sbd_b);
        except(sbd_err == CL_SUCCESS, "failed to set kernel arguments");

        sbd_err = clEnqueueNDRangeKernel(sbd_queue, originalLoop_kernel, 2, NULL, global_work,local_work,0,NULL,NULL);
        except(sbd_err == CL_SUCCESS, "failed to execute kernel");
        clFinish(sbd_queue);
        
        LSB_Rec(i);
        
        LSB_Set_Rparam_string("region","device_side_d2h_copy");
        LSB_Res();
        sbd_err  = clEnqueueReadBuffer(sbd_queue,sbd_a,CL_TRUE,0,a_bytes,a,0,NULL,NULL);
        sbd_err |= clEnqueueReadBuffer(sbd_queue,sbd_b,CL_TRUE,0,b_bytes,b,0,NULL,NULL);
        except(sbd_err == CL_SUCCESS, "can't read from device memory");
        LSB_Rec(i);
    }
    //loop blocking case
    LSB_Set_Rparam_string("kernel","transformed_loop_after_blocking");
    for(int i = 0; i < sample_size; i++){
        LSB_Set_Rparam_string("region", "host_side_initialise_buffers");
        randomise_payload(a,M*M);
        randomise_payload(b,M*M);
        LSB_Rec(i);

        LSB_Set_Rparam_string("region","device_side_h2d_copy");
        LSB_Res();
        sbd_err  = clEnqueueWriteBuffer(sbd_queue,sbd_a,CL_TRUE,0,a_bytes,a,0,NULL,NULL);
        sbd_err |= clEnqueueWriteBuffer(sbd_queue,sbd_b,CL_TRUE,0,b_bytes,b,0,NULL,NULL);
        except(sbd_err == CL_SUCCESS, "can't write to device memory!");
        LSB_Rec(i);

        //run the kernel
        size_t global_work[2] = {static_cast<size_t>(M),static_cast<size_t>(M)};
        //size_t local_work[2] = {static_cast<size_t>(w),static_cast<size_t>(w)}; 
        //on one thread
        size_t local_work[2] = {static_cast<size_t>(M),static_cast<size_t>(M)}; 
        LSB_Set_Rparam_string("region","transformed_loop_after_blocking_kernel");
        LSB_Res();
        sbd_err  = clSetKernelArg(transformedLoop_kernel, 0, sizeof(cl_mem), &sbd_a);
        sbd_err |= clSetKernelArg(transformedLoop_kernel, 1, sizeof(cl_mem), &sbd_b);
        except(sbd_err == CL_SUCCESS, "failed to set kernel arguments");

        sbd_err = clEnqueueNDRangeKernel(sbd_queue, transformedLoop_kernel, 2, NULL, global_work,local_work,0,NULL,NULL);

        except(sbd_err == CL_SUCCESS, "failed to execute kernel");

        clFinish(sbd_queue);
        LSB_Rec(i);
        
        LSB_Set_Rparam_string("region","device_side_d2h_copy");
        LSB_Res();
        sbd_err  = clEnqueueReadBuffer(sbd_queue,sbd_a,CL_TRUE,0,a_bytes,a,0,NULL,NULL);
        sbd_err |= clEnqueueReadBuffer(sbd_queue,sbd_b,CL_TRUE,0,b_bytes,b,0,NULL,NULL);
        except(sbd_err == CL_SUCCESS, "can't read from device memory");
        LSB_Rec(i);
    }
    //print_payload(a,M*M);
    delete a;
    delete b;

    clReleaseMemObject(sbd_b);
    clReleaseMemObject(sbd_a);
    clReleaseKernel(transformedLoop_kernel);
    clReleaseKernel(originalLoop_kernel);
    clReleaseProgram(sbd_program);
    clReleaseCommandQueue(sbd_queue);
    clReleaseContext(sbd_context);
    
    delete sk_source;
    delete sbd_devices;
    delete sbd_platforms;
    LSB_Finalize();
}

