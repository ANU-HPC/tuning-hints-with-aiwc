#include <iostream>
#include <cstring>
#include <cassert>
#include <fstream>
#include <ctime>
#include <cstdlib>
#include <random>
#include <cstdio>
#include <liblsb.h>
extern "C" {
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
}
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


inline void identity(float* x, unsigned int size, int dim) {
    for (int i = 0; i < size; i++) {
        x[i] = (i % (dim+1)) ? 0 : 1;
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
    int dim = (int)(sqrt((float)size));
    for(int i = 0; i < size; i++){
        std::cout << x[i] << ' ';
        if (!(i % dim))
            std::cout << std::endl;
    }
    std::cout << std::endl;
}

inline void check_matrix_multiply(float* a, float* b, float* c, size_t M) {
    gsl_matrix_float* a_matrix = gsl_matrix_float_alloc(M, M);
    gsl_matrix_float* b_matrix = gsl_matrix_float_alloc(M, M);
    gsl_matrix_float* c_matrix = gsl_matrix_float_alloc(M, M);
    for (int j = 0; j < M; j++) {
        for (int k = 0; k < M; k++) {
            gsl_matrix_float_set(a_matrix, j,k, (double) a[j*M + k]);
            gsl_matrix_float_set(b_matrix, j,k, (double) b[j*M + k]);
        }
    }
    gsl_blas_sgemm(CblasNoTrans, CblasNoTrans, 1.0, a_matrix, b_matrix, 0.0, c_matrix); 
    for (int j = 0; j < M; j++) {
        for (int k = 0; k < M; k++) {
            except (c[j*M + k] - gsl_matrix_float_get(c_matrix, j, k) > (double)EPSILON 
                || c[j*M + k] - gsl_matrix_float_get(c_matrix, j, k) < -(double)EPSILON,
                "matrix multiplication algorithm is wrong");
        }
    }
}

int main(int argc, char** argv){

    //extract kernel -- ./sbd <kernel source> <tiny/small/medium/large> <platform id> <device id>
    except(argc == 6, "./sbd <kernel source> <tiny/small/medium/large> <platform id> <device id> <aiwc/runtime>");
    char* synthetic_kernel_path = argv[1];
    char* problem_size = argv[2];
    int platform_id = atoi(argv[3]);
    int device_id = atoi(argv[4]);
    char* mode = argv[5];

    LSB_Init("gpu_memory_access",0);
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
    if(strcmp(problem_size, "tiny")==0)       {KiB = 32 ;}    //  32 KiB < L1
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
    int N = M;//just use even sized matrices to keep it simple

    c_bytes = M*N*sizeof(float);
    unsigned int a_bytes = w*M*sizeof(float);
    unsigned int b_bytes = w*N*sizeof(float);


    std::cout << "M = " << M << " N = " << N << " total KiB = " <<  (c_bytes+a_bytes+b_bytes)/1024 << std::endl;
    
    LSB_Rec(0);

    

    std::cout << "Operating on a " << M << "x" << M << " matrix with a tile size " << w << "..." << std::endl;

    LSB_Set_Rparam_string("region", "kernel_creation");
    LSB_Res();
    //compile kernels
    std::string compiler_flags = "-DTILE_DIM=" + std::to_string(w); 
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
    cl_kernel simpleMultiply_kernel = clCreateKernel(sbd_program, "simpleMultiply", &sbd_err);
    except(sbd_err == CL_SUCCESS, "can't create kernel");
    cl_kernel coalescedMultiply_kernel = clCreateKernel(sbd_program, "coalescedMultiply", &sbd_err);
    cl_kernel sharedABMultiply_kernel= clCreateKernel(sbd_program, "sharedABMultiply", &sbd_err);
    except(sbd_err == CL_SUCCESS, "can't create kernel");
    LSB_Rec(0);

    //memory setup
    LSB_Set_Rparam_string("region", "device_side_buffer_setup");
    LSB_Res();
    cl_mem sbd_a = clCreateBuffer(sbd_context,CL_MEM_READ_WRITE,a_bytes,NULL,&sbd_err);
    except(sbd_err == CL_SUCCESS, "can't create device memory a");
    cl_mem sbd_b = clCreateBuffer(sbd_context,CL_MEM_READ_WRITE,b_bytes,NULL,&sbd_err);
    except(sbd_err == CL_SUCCESS, "can't create device memory b");
    cl_mem sbd_c = clCreateBuffer(sbd_context,CL_MEM_READ_WRITE,c_bytes,NULL,&sbd_err);
    except(sbd_err == CL_SUCCESS, "can't create device memory c");

    float* a  = new float[M*w](); 
    float* b  = new float[N*w]();
    float* c  = new float[M*N]();
    LSB_Rec(0);

    int sample_size = 100;
    if(strcmp(mode, "aiwc")==0){
        sample_size = 1;
    }
    //simple case
    LSB_Set_Rparam_string("kernel","simpleMultiply");
    for(int i = 0; i < sample_size; i++){
        LSB_Set_Rparam_string("region", "host_side_initialise_buffers");
        randomise_payload(a,M*w);
        identity(b,N*w, N);
        print_payload(b, N*w);
        randomise_payload(c,M*N);
        // for (int t = 0; t < M; t++) {
        //     for (int u = 0; u < M; u++) {
        //         printf("%f ", a[t*M + u]);
        //     }
        //     printf("\n");
        // }
    
        gsl_matrix* a_matrix = gsl_matrix_alloc(M, N);
        gsl_matrix* b_matrix = gsl_matrix_alloc(M, N);
        gsl_matrix* c_matrix = gsl_matrix_alloc(M, N);
        for (int j = 0; j < M; j++) {
            for (int k = 0; k < N; k++) {
                gsl_matrix_set(a_matrix, j, k, (double) a[j*N + k]);
                gsl_matrix_set(b_matrix, j, k, (double) b[j*N + k]);
            }
        }
        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, a_matrix, b_matrix, 0.0, c_matrix); 
        LSB_Rec(i);

        LSB_Set_Rparam_string("region","device_side_h2d_copy");
        LSB_Res();
        sbd_err  = clEnqueueWriteBuffer(sbd_queue,sbd_a,CL_TRUE,0,a_bytes,a,0,NULL,NULL);
        sbd_err |= clEnqueueWriteBuffer(sbd_queue,sbd_b,CL_TRUE,0,b_bytes,b,0,NULL,NULL);
        sbd_err |= clEnqueueWriteBuffer(sbd_queue,sbd_c,CL_TRUE,0,c_bytes,c,0,NULL,NULL);
        except(sbd_err == CL_SUCCESS, "can't write to device memory!");
        LSB_Rec(i);

        //run the kernel
        size_t global_work[2] = {static_cast<size_t>(M),static_cast<size_t>(M)};
        size_t local_work[2] = {static_cast<size_t>(w),static_cast<size_t>(w)}; 

        LSB_Set_Rparam_string("region","simpleMultiply_kernel");
        LSB_Res();
        sbd_err = clSetKernelArg(simpleMultiply_kernel, 0, sizeof(cl_mem), &sbd_a);
        sbd_err = clSetKernelArg(simpleMultiply_kernel, 1, sizeof(cl_mem), &sbd_b);
        sbd_err = clSetKernelArg(simpleMultiply_kernel, 2, sizeof(cl_mem), &sbd_c);
        sbd_err = clSetKernelArg(simpleMultiply_kernel, 3, sizeof(cl_int), &M);
        except(sbd_err == CL_SUCCESS, "failed to set kernel arguments");

        sbd_err = clEnqueueNDRangeKernel(sbd_queue, simpleMultiply_kernel, 2, NULL, global_work,local_work,0,NULL,NULL);
        except(sbd_err == CL_SUCCESS, "failed to execute kernel");

        clFinish(sbd_queue);
        LSB_Rec(i);
        
        LSB_Set_Rparam_string("region","device_side_d2h_copy");
        LSB_Res();
        //FIXME: Don't see why we must copy a and b back d2h. If necessary, pls uncomment
        //sbd_err  = clEnqueueReadBuffer(sbd_queue,sbd_a,CL_TRUE,0,a_bytes,a,0,NULL,NULL);
        //sbd_err |= clEnqueueReadBuffer(sbd_queue,sbd_b,CL_TRUE,0,b_bytes,b,0,NULL,NULL);
        sbd_err |= clEnqueueReadBuffer(sbd_queue,sbd_c,CL_TRUE,0,c_bytes,c,0,NULL,NULL);
        except(sbd_err == CL_SUCCESS, "can't read from device memory");
        LSB_Rec(i);
        // c[256] = 17.0;
        except(same_payload(a,c,N*w), "a, c not equal");
        for (int j = 0; j < M; j++)
            for (int k = 0; k < N; k++) {
                if (c[j*N + k] - gsl_matrix_get(c_matrix, j, k) > (double)EPSILON 
                 || c[j*N + k] - gsl_matrix_get(c_matrix, j, k) < -(double)EPSILON) {
                    printf("matrix multiplication algorithm is wrong %d %d", j, k);
                    
                    except(false, "matrix multiply wrong");
                 }
            }
    }
    //coalescedMultiply case
    LSB_Set_Rparam_string("kernel","coalescedMultiply");
    for(int i = 0; i < sample_size; i++){
        LSB_Set_Rparam_string("region", "host_side_initialise_buffers");
        randomise_payload(a,M*w);
        randomise_payload(b,N*w);
        randomise_payload(c,M*N);
        LSB_Rec(i);

        LSB_Set_Rparam_string("region","device_side_h2d_copy");
        LSB_Res();
        sbd_err  = clEnqueueWriteBuffer(sbd_queue,sbd_a,CL_TRUE,0,a_bytes,a,0,NULL,NULL);
        sbd_err |= clEnqueueWriteBuffer(sbd_queue,sbd_b,CL_TRUE,0,b_bytes,b,0,NULL,NULL);
        sbd_err |= clEnqueueWriteBuffer(sbd_queue,sbd_c,CL_TRUE,0,c_bytes,c,0,NULL,NULL);
        except(sbd_err == CL_SUCCESS, "can't write to device memory!");
        LSB_Rec(i);

        //run the kernel
        size_t global_work[2] = {static_cast<size_t>(M),static_cast<size_t>(M)};
        size_t local_work[2] = {static_cast<size_t>(w),static_cast<size_t>(w)}; 

        LSB_Set_Rparam_string("region","coalescedMultiply_kernel");
        LSB_Res();
        sbd_err = clSetKernelArg(coalescedMultiply_kernel, 0, sizeof(cl_mem), &sbd_a);
        sbd_err = clSetKernelArg(coalescedMultiply_kernel, 1, sizeof(cl_mem), &sbd_b);
        sbd_err = clSetKernelArg(coalescedMultiply_kernel, 2, sizeof(cl_mem), &sbd_c);
        sbd_err = clSetKernelArg(coalescedMultiply_kernel, 3, sizeof(cl_int), &w);
        except(sbd_err == CL_SUCCESS, "failed to set kernel arguments");

        sbd_err = clEnqueueNDRangeKernel(sbd_queue, coalescedMultiply_kernel, 2, NULL, global_work,local_work,0,NULL,NULL);
        except(sbd_err == CL_SUCCESS, "failed to execute kernel");

        clFinish(sbd_queue);
        LSB_Rec(i);
        
        LSB_Set_Rparam_string("region","device_side_d2h_copy");
        LSB_Res();
        sbd_err  = clEnqueueReadBuffer(sbd_queue,sbd_a,CL_TRUE,0,a_bytes,a,0,NULL,NULL);
        sbd_err |= clEnqueueReadBuffer(sbd_queue,sbd_b,CL_TRUE,0,b_bytes,b,0,NULL,NULL);
        sbd_err |= clEnqueueReadBuffer(sbd_queue,sbd_c,CL_TRUE,0,c_bytes,c,0,NULL,NULL);
        except(sbd_err == CL_SUCCESS, "can't read from device memory");
        LSB_Rec(i);
    }
    //sharedABMultiply
    LSB_Set_Rparam_string("kernel","sharedABMultiply");
    for(int i = 0; i < sample_size; i++){
        LSB_Set_Rparam_string("region", "host_side_initialise_buffers");
        randomise_payload(a,M*w);
        randomise_payload(b,N*w);
        randomise_payload(c,M*N);
        LSB_Rec(i);

        LSB_Set_Rparam_string("region","device_side_h2d_copy");
        LSB_Res();
        sbd_err  = clEnqueueWriteBuffer(sbd_queue,sbd_a,CL_TRUE,0,a_bytes,a,0,NULL,NULL);
        sbd_err |= clEnqueueWriteBuffer(sbd_queue,sbd_b,CL_TRUE,0,b_bytes,b,0,NULL,NULL);
        sbd_err |= clEnqueueWriteBuffer(sbd_queue,sbd_c,CL_TRUE,0,c_bytes,c,0,NULL,NULL);
        except(sbd_err == CL_SUCCESS, "can't write to device memory!");
        LSB_Rec(i);

        //run the kernel
        size_t global_work[2] = {static_cast<size_t>(M),static_cast<size_t>(M)};
        size_t local_work[2] = {static_cast<size_t>(w),static_cast<size_t>(w)}; 

        LSB_Set_Rparam_string("region","sharedABMultiply_kernel");
        LSB_Res();
        sbd_err = clSetKernelArg(sharedABMultiply_kernel, 0, sizeof(cl_mem), &sbd_a);
        sbd_err = clSetKernelArg(sharedABMultiply_kernel, 1, sizeof(cl_mem), &sbd_b);
        sbd_err = clSetKernelArg(sharedABMultiply_kernel, 2, sizeof(cl_mem), &sbd_c);
        sbd_err = clSetKernelArg(sharedABMultiply_kernel, 3, sizeof(cl_int), &w);
        except(sbd_err == CL_SUCCESS, "failed to set kernel arguments");

        sbd_err = clEnqueueNDRangeKernel(sbd_queue, sharedABMultiply_kernel, 2, NULL, global_work,local_work,0,NULL,NULL);
        except(sbd_err == CL_SUCCESS, "failed to execute kernel");

        clFinish(sbd_queue);
        LSB_Rec(i);
        
        LSB_Set_Rparam_string("region","device_side_d2h_copy");
        LSB_Res();
        sbd_err  = clEnqueueReadBuffer(sbd_queue,sbd_a,CL_TRUE,0,a_bytes,a,0,NULL,NULL);
        sbd_err |= clEnqueueReadBuffer(sbd_queue,sbd_b,CL_TRUE,0,b_bytes,b,0,NULL,NULL);
        sbd_err |= clEnqueueReadBuffer(sbd_queue,sbd_c,CL_TRUE,0,c_bytes,c,0,NULL,NULL);
        except(sbd_err == CL_SUCCESS, "can't read from device memory");
        LSB_Rec(i);
    }
    //print_payload(a,M*M);
    delete a;
    delete b;
    delete c;

    clReleaseMemObject(sbd_c);
    clReleaseMemObject(sbd_b);
    clReleaseMemObject(sbd_a);
    clReleaseKernel(sharedABMultiply_kernel);
    clReleaseKernel(coalescedMultiply_kernel);
    clReleaseKernel(simpleMultiply_kernel);
    clReleaseProgram(sbd_program);
    clReleaseCommandQueue(sbd_queue);
    clReleaseContext(sbd_context);
    
    delete sk_source;
    delete sbd_devices;
    delete sbd_platforms;
    LSB_Finalize();
}


