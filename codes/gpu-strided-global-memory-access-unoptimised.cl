//shared memory in matrix multiplication ported from the [cuda c best practices guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-in-matrix-multiplication-c-aa)

__kernel void simpleMultiply(__global float *a,__global float *b,__global float *c, int N)
{
    int row = get_group_id(1) * get_local_size(1) + get_local_id(1);//blockIdx.y * blockDim.y + threadIdx.y;
    int col = get_group_id(0) * get_local_size(0) + get_local_id(0);//blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for (int i = 0; i < TILE_DIM; i++) {
        sum += a[row*TILE_DIM+i] * b[i*N+col];
    }
    c[row*TILE_DIM+col] = sum;
}

__kernel void coalescedMultiply(__global float* a, __global float* b, __global float* c, int N)
{
    __local float aTile[TILE_DIM][TILE_DIM];
    int row = get_group_id(1) * get_local_size(1) + get_local_id(1);//blockIdx.y * blockDim.y + threadIdx.y;
    int col = get_group_id(0) * get_local_size(0) + get_local_id(0);//blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    aTile[get_local_id(1)][get_local_id(0)] = a[row*TILE_DIM+get_local_id(0)];
    for (int i = 0; i < TILE_DIM; i++) {
        sum += aTile[get_local_id(1)][i] * b[i*N+col];
    }
    c[row*N+col] = sum;
}

__kernel void sharedABMultiply(__global float *a, __global float* b, __global float *c, int N)
{
    __local float aTile[TILE_DIM][TILE_DIM],
                  bTile[TILE_DIM][TILE_DIM];
    int row = get_group_id(1) * get_local_size(1) + get_local_id(1);
    int col = get_group_id(0) * get_local_size(0) + get_local_id(0);
    float sum = 0.0f;
//    aTile[get_local_id(1)][get_local_id(0)] = a[row*TILE_DIM+get_local_id(0)];
//    bTile[get_local_id(1)][get_local_id(0)] = b[get_local_id(1)*N+col];
//    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = 0; i < TILE_DIM; i++) {
        sum += aTile[get_local_id(1)][i]* bTile[i][get_local_id(0)];
    }
    c[row*N+col] = sum;
}


