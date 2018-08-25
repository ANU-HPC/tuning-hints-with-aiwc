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
    __local float bTile[TILE_DIM][TILE_DIM];

    int row = get_group_id(1) * get_local_size(1) + get_local_id(1);//blockIdx.y * blockDim.y + threadIdx.y;
    int col = get_group_id(0) * get_local_size(0) + get_local_id(0);//blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    int x = get_local_id(0);
    int y = get_local_id(1);

    aTile[y][x] = a[row*TILE_DIM+x];
    bTile[y][x] = b[y*N+col];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = 0; i < TILE_DIM; i++) {
        sum += aTile[y][i]* bTile[i][x];
    }
    c[row*N+col] = sum;
}


__kernel void transposedCoalescedMultiply(__global float* a, __global float* b, __global float* c, int N)
{
    __local float aTile[TILE_DIM][TILE_DIM];
    __local float transposedTile[TILE_DIM][TILE_DIM];

    int row = get_group_id(1) * get_local_size(1) + get_local_id(1);//blockIdx.y * blockDim.y + threadIdx.y;
    int col = get_group_id(0) * get_local_size(0) + get_local_id(0);//blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    int x = get_local_id(0);
    int y = get_local_id(1);

    aTile[y][x] = a[row*TILE_DIM+x];
    transposedTile[x][y] = a[(get_group_id(0)*get_local_size(0) + get_local_id(1))*TILE_DIM + get_local_id(0)];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = 0; i < TILE_DIM; i++) {
        sum += aTile[y][i]*transposedTile[i][x];
    }
    c[row*N+col] = sum;
}


