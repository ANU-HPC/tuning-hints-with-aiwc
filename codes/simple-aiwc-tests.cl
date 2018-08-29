
__kernel void dummyStore(__global float *a,__global float *b,__global float *c, int N)
{
    int x = get_group_id(0) * get_local_size(0) + get_local_id(0);//blockIdx.x * blockDim.x + threadIdx.x;
    c[x] = x;
}

__kernel void doubleStore(__global float *a,__global float *b,__global float *c, int N)
{
    int x = get_group_id(0) * get_local_size(0) + get_local_id(0);//blockIdx.x * blockDim.x + threadIdx.x;
    c[x] = x;
    b[x] = x;
}

__kernel void simpleStore(__global float *a,__global float *b,__global float *c, int N)
{
    int x = get_group_id(0) * get_local_size(0) + get_local_id(0);//blockIdx.x * blockDim.x + threadIdx.x;
    c[x] = a[x];
}

__kernel void doubleRead(__global float *a,__global float *b,__global float *c, int N)
{
    int x = get_group_id(0) * get_local_size(0) + get_local_id(0);//blockIdx.x * blockDim.x + threadIdx.x;
    c[x] = a[x] + b[x];
}


