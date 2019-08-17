//shared memory in matrix multiplication ported from the [cuda c best practices guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-in-matrix-multiplication-c-aa)


__kernel void simpleMultiply(__global float *a,__global float *b,__global float *c, int N)
{
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)
 
    // Compute a single element (loop over K)
    float acc = 0.0f;
    for (int k=0; k<N; k++) {
        acc += b[k*N + globalCol] * a[globalRow*N + k];
    }
 
    // Store the result
    c[globalRow*N + globalCol] = acc;
}

__kernel void coalescedMultiply(const __global float* A, 
                                const __global float* B,
                                __global float* C, const int N)
{
    __local float aTile[TILE_DIM][TILE_DIM];

    const int localRow = get_local_id(0);
    const int localCol = get_local_id(1);

    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);
    //int row = get_group_id(1) * get_local_size(1) + get_local_id(1);//blockIdx.y * blockDim.y + threadIdx.y; //get_global_id()
    //int col = get_group_id(0) * get_local_size(0) + get_local_id(0);//blockIdx.x * blockDim.x + threadIdx.x; //get_global_id()
    __private float sum = 0.0f;

    const int numTiles = N / TILE_DIM;
    for (int i = 0; i < numTiles; i++) {
        const int tiledRow = globalRow*N+i*TILE_DIM + localCol;
        aTile[localRow][localCol] = A[tiledRow];
        //aTile[localRow][localCol] = A[globalRow*N+localCol+i*TILE_DIM];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < TILE_DIM; k++) {
            sum += aTile[localRow][k] * B[(i*TILE_DIM+k)*N+globalCol];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
    }
    C[globalRow*N+globalCol] = sum;
}

__kernel void sharedABMultiply(__global float *A, __global float* B, __global float *C, int N)
{
    __local float aTile[TILE_DIM][TILE_DIM],
                  bTile[TILE_DIM][TILE_DIM];
    
    barrier(CLK_LOCAL_MEM_FENCE);

    const int localRow = get_local_id(0);
    const int localCol = get_local_id(1);

    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);
    
    float sum = 0.0f;
    const int numTiles = N / TILE_DIM;

    for (int i = 0; i < numTiles; i++) {
    //    aTile[localRow][localCol] = A[row*TILE_DIM+get_local_id(0)];
    //    bTile[localRow][localCol] = B[get_local_id(1)*N+col];
    //    barrier(CLK_LOCAL_MEM_FENCE);
        const int tiledRow = globalRow*N+i*TILE_DIM + localCol;
        const int tiledCol = globalCol + (TILE_DIM*i + localRow)*N;
        aTile[localRow][localCol] = A[tiledRow];
        bTile[localCol][localRow] = B[tiledCol];

        // aTile[localRow][localCol] = A[globalRow*N+localCol+i*TILE_DIM];
        // bTile[localRow][localCol] = B[localRow + TILE_DIM*get_group_id(1) 
        //                             + N*(TILE_DIM*get_group_id(0) + localCol)]; // Implicit transpose
        barrier(CLK_LOCAL_MEM_FENCE);                
        for (int k = 0; k < TILE_DIM; k++) {
            sum += aTile[localRow][k]* bTile[localCol][k]; //???
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    C[globalRow*N+globalCol] = sum;
}

/*
OpenCL Kernels for matrix multiplicatioglobalRow*N+i*TILE_DIM + localColby Cedric Nugeteren

    The MIT License (MIT)

Copyright (c) 2014 SURFsara

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/*
    Coalesced and tiled calculations -- splitting the M*N matrix
    into several k*k workgroups and calculating smartly after
    coalescing into shared memory
*/
//__kernel void GEMM2(__global float *a,__global float *b,__global float *c, int N)
// __kernel void GEMM2(  const __global float* A,
//                       const __global float* B,
//                       __global float* C,
//                       const int N) {
    
//     // Thread identifiers
//     const int row = get_local_id(0); // Local row ID (max: TS)
//     const int col = get_local_id(1); // Local col ID (max: TS)
//     const int globalRow = TILE_DIM*get_group_id(0) + row; // Row ID of C (0..M)
//     const int globalCol = TILE_DIM*get_group_id(1) + col; // Col ID of C (0..N)
 
//     // Local memory to fit a tile of TS*TS elements of A and B
//     __local float Asub[TILE_DIM][TILE_DIM];
//     __local float Bsub[TILE_DIM][TILE_DIM];
 
//     // Initialise the accumulation register
//     float acc = 0.0f;
    
//     // Loop over all tiles
//     const int numTiles = N/TILE_DIM;
//     for (int t=0; t<numTiles; t++) {
 
//         // Load one tile of A and B into local memory
//         const int tiledRow = TILE_DIM*t + row;
//         const int tiledCol = TILE_DIM*t + col;
//         //Asub[col][row] = A[globalCol*N + tiledRow];
//         //Bsub[col][row] = B[tiledCol*N + globalRow];

//         Asub[col][row] = A[tiledCol*N + globalRow];
//         Bsub[col][row] = B[globalCol*N + tiledRow];
 
//         // Synchronise to make sure the tile is loaded
//         barrier(CLK_LOCAL_MEM_FENCE);
 
//         // Perform the computation for a single tile
//         for (int k=0; k<TILE_DIM; k++) {
//             //acc += Asub[col][k] * Bsub[k][row];
//             acc += Asub[k][row] * Bsub[col][k];

//         }
 
//         // Synchronise before loading the next tile
//         barrier(CLK_LOCAL_MEM_FENCE);
//     }
//     // Store the final result in C32
//     //C[globalCol + globalRow*N] = acc;32
//     C[globalCol*N + globalRow] = acc;
// }

__kernel void GEMM2Multiply(const __global float* A,
                      const __global float* B,
                      __global float* C,
                      const int N) {


    const int localRow = get_local_id(0);
    const int localCol = get_local_id(1);

    const int globalRow = localRow + get_group_id(0) * get_local_size(0);
    const int globalCol = localCol + get_group_id(1) * get_local_size(1);

    __local float ASub[TILE_DIM][TILE_DIM];
    __local float BSub[TILE_DIM][TILE_DIM];

    barrier(CLK_LOCAL_MEM_FENCE);

    float acc = 0.0f;

    const int numTiles = N/TILE_DIM;
    for (int i = 0; i < numTiles; i++) {
        const int tiledRow = globalRow*N+i*TILE_DIM + localCol;
        const int tiledCol = globalCol + (TILE_DIM*i + localRow)*N;
        ASub[localRow][localCol] = A[tiledRow];
        BSub[localRow][localCol] = B[tiledCol];

        barrier(CLK_LOCAL_MEM_FENCE);
    
        for (int k=0; k<TILE_DIM; k++) {
            //acc += Asub[col][k] * Bsub[k][row];
            acc += ASub[localRow][k] * BSub[k][localCol];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
    C[globalRow*N + globalCol] = acc;
}

/*
__kernel void GEMM2(const int M, const int N, const int K,
                      const __global float* A,
                      const __global float* B,
                      __global float* C) {
    
    // Thread identifiers
    const int row = get_local_id(0); // Local row ID (max: TS)
    const int col = get_local_id(1); // Local col ID (max: TS)
    const int globalRow = TS*get_group_id(0) + row; // Row ID of C (0..M)
    const int globalCol = TS*get_group_id(1) + col; // Col ID of C (0..N)
 
    // Local memory to fit a tile of TS*TS elements of A and B
    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];
 
    // Initialise the accumulation register
    float acc = 0.0f;
    
    // Loop over all tiles
    const int numTiles = K/TS;
    for (int t=0; t<numTiles; t++) {
 
        // Load one tile of A and B into local memory
        const int tiledRow = TS*t + row;
        const int tiledCol = TS*t + col;
        Asub[col][row] = A[tiledCol*M + globalRow];
        Bsub[col][row] = B[globalCol*K + tiledRow];
 
        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // Perform the computation for a single tile
        for (int k=0; k<TS; k++) {
            acc += Asub[k][row] * Bsub[col][k];
        }
 
        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
 
    // Store the final result in C
    C[glo
*/


/*
    More work per thread: by factor WPT.
*/
/*
__kernel void myGEMM3(const __global float* A,
                      const __global float* B,
                      __global float* C, const int N) {
    
    // Thread identifiers
    const int row = get_local_id(0); // Local row ID (max: TS)
    const int col = get_local_id(1); // Local col ID (max: TS/WPT == RTS)
    const int globalRow = TILE_DIM*get_group_id(0) + row; // Row ID of C (0..M)
    const int globalCol = TILE_DIM*get_group_id(1) + col; // Col ID of C (0..N)
 
    // Local memory to fit a tile of TS*TS elements of A and B
    __local float Asub[TILE_DIM][TILE_DIM];
    __local float Bsub[TILE_DIM][TILE_DIM];
 
    // Initialise the accumulation registers
    float acc[WPT];
    for (int w=0; w<WPT; w++) {
        acc[w] = 0.0f;
    }
    
    // Loop over all tiles
    const int numTiles = N/TILE_DIM;
    for (int t=0; t<numTiles; t++) {
 
        // Load one tile of A and B into local memory
        for (int w=0; w<WPT; w++) {
            const int tiledRow = TILE_DIM*t + row;
            const int tiledCol = T*t + col;
            Asub[col + w*RTS][row] = A[(tiledCol + w*RTS)*M + globalRow];
            Bsub[col + w*RTS][row] = B[(globalCol + w*RTS)*K + tiledRow];
        }
        
        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // Perform the computation for a single tile
        for (int k=0; k<TS; k++) {
            for (int w=0; w<WPT; w++) {
                acc[w] += Asub[k][row] * Bsub[col + w*RTS][k];
            }
        }
 
        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
 
    // Store the final results in C
    for (int w=0; w<WPT; w++) {
        C[(globalCol + w*RTS)*M + globalRow] = acc[w];
    }
}
*/

/*
Wider data access instructions
*/