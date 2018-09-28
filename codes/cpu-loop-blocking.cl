//Loop Blocking Example 4-25 from https://www.intel.com/content/dam/www/public/us/en/documents/manuals/64-ia-32-architectures-optimization-manual.pdf in Section 4.5.3 Loop Blocking
//TODO: verify loop-blocking correctness
__kernel void original_loop(__global float* A, __global float* B)
{
    //get_group_id(0) * get_local_size(0) + get_local_id(0);//not needed -- intel tests only for single thread
    for (int i=0; i< MAX; i++) {
        for (int j=0; j< MAX; j++) {
            //A[i,j] = A[i,j] + B[j, i];
            A[i*MAX+j] = A[i*MAX+j] + B[j*MAX+i];
        } 
    }
}

__kernel void transformed_loop_after_blocking(__global float* A, __global float* B)
{
    int i,j,ii,jj;
    for (i=0; i< MAX; i+=BLOCK_SIZE) {
        for (j=0; j< MAX; j+=BLOCK_SIZE) {
            for (ii=i; ii<i+BLOCK_SIZE; ii++) {
                for (jj=j; jj<j+BLOCK_SIZE; jj++) {
                    //A[ii,jj] = A[ii,jj] + B[jj, ii];
                    A[ii*MAX+jj] = A[ii*MAX+jj] + B[jj*MAX+ii];
                }
            }
        }
    }
}

