//Example 5-32 Baseline C Code for Mandelbrot Set Map Evaluation from:https://www.intel.com/content/dam/www/public/us/en/documents/manuals/64-ia-32-architectures-optimization-manual.pdf

//#define DIMX (64)
//#define DIMY (64)
//#define X_STEP (0.5f/DIMX)
//#define Y_STEP (0.4f/(DIMY/2))
__kernel void mandelbrot(__global int* map)
{
    int i,j;
    float x = -1.8f;
    for (i=0;i<DIMX;i++) {
        float y = -0.2f;
        for (j=0;j<DIMY/2;j++) {
            int iter = 0;
            float sx = x;
            float sy = y;
            while (iter < 256){
                if (sx*sx + sy*sy >= 4.0f){
                    break;
                }
                float old_sx = sx;
                sx = x + sx*sx - sy*sy;
                sy = y + 2*old_sx*sy;
                iter++;
            }
            map[i*DIMY+j] = iter;
            y+=Y_STEP;
        }
        x+=X_STEP;
    }
}

//Blend packed single-precision (32-bit) floating-point elements from a and b using mask, and store the results in dest.
inline float4 blend(float4 a, float4 b, int4 mask){
    float4 dest;
    if(mask.x){
        dest.x = b.x;
    }else{
        dest.x = a.x;
    }
    if(mask.y){
        dest.y = b.y;
    }else{
        dest.y = a.y;
    }
    if(mask.z){
        dest.z = b.z;
    }else{
        dest.z = a.z;
    }
    if(mask.w){
        dest.w = b.w;
    }else{
        dest.w = a.w;
    }
    return(dest);
}

inline int4 blend_ints(int4 a, int4 b, int4 mask){
    int4 dest;
    if(mask.x){
        dest.x = b.x;
    }else{
        dest.x = a.x;
    }
    if(mask.y){
        dest.y = b.y;
    }else{
        dest.y = a.y;
    }
    if(mask.z){
        dest.z = b.z;
    }else{
        dest.z = a.z;
    }
    if(mask.w){
        dest.w = b.w;
    }else{
        dest.w = a.w;
    }
    return(dest);
}

inline int allones(int4 a){
    if(a.x == 1 && a.y == 1 && a.z == 1 && a.w == 1){
        return (1);
    }else{
        return(0);
    }
}

inline int allzeros(int4 a){
    if(a.x == 0 && a.y == 0 && a.z == 0 && a.w == 0){
        return (1);
    }else{
        return(0);
    }
}

__kernel void mandelbrot_vectorized(__global int* map)
{
    //Example 5-33 Vectorized Mandelbrot Set Map Evaluation Using SSE4.1 Intrinsics from:https://www.intel.com/content/dam/www/public/us/en/documents/manuals/64-ia-32-architectures-optimization-manual.pdf
    const float4 xstep = (float4)(X_STEP,X_STEP,X_STEP,X_STEP);
    const float4 ystep = (float4)(Y_STEP,Y_STEP,Y_STEP,Y_STEP);
    const float4 four_step = (float4)(Y_STEP*4,Y_STEP*4,Y_STEP*4,Y_STEP*4);
    const float4 init_y = (float4)(0,1*Y_STEP,2*Y_STEP,3*Y_STEP);
    const float4 four_vec = (float4)(4.0,4.0,4.0,4.0);
    const float4 two_vec = (float4)(2.0,2.0,2.0,2.0);
    const int4 zero_ivec = (int4)(0,0,0,0);
    const int4 one_ivec = (int4)(1,1,1,1);

    float4 x = (float4)(-1.8f,-1.8f,-1.8f,-1.8f);
    for (int i=0;i<DIMX;i++) {
        float4 y = (float4)(-0.2f,-0.2f,-0.2f,-0.2f)+init_y;
        for (int j = 0; j < DIMY/2; j+=4) {
            int4 iter = zero_ivec;
            float4 sx = x;
            float4 sy = y;
            int scalar_iter = 0;
            while (scalar_iter < 256){
                float4 old_sx = sx;
                int4 vmask = isgreaterequal(sx*sx + sy*sy, four_vec);
                //if (allones(vmask)){
                if (all(vmask)){
                    break;
                }
                //if (allzeros(vmask)){
                //if (all(isequal(vmask,(int4)(0,0,0,0)))){//changed this since its only defined for floats (but intel allowed it)
                if (all(vmask==(int4)(0,0,0,0))){
                    sx = x + sx*sx - sy*sy;
                    sy = y + two_vec*old_sx*sy;
                    iter += one_ivec;
                }else{
                    //sx = blend(x+sx*sx-sy*sy,sx,vmask);
                    //sy = blend(y+two_vec*old_sx*sy,sy,vmask);
                    //iter = blend_ints(iter+one_ivec,iter,vmask);
                    sx = select(x+sx*sx-sy*sy,sx,vmask);
                    sy = select(y+two_vec*old_sx*sy,sy,vmask);
                    iter = select(iter+one_ivec,iter,vmask);
                }
                scalar_iter++;
            }

            //map[i*DIMY+j+0] = iter.x;
            //map[i*DIMY+j+1] = iter.y;
            //map[i*DIMY+j+2] = iter.z;
            //map[i*DIMY+j+3] = iter.w;
            vstore4(iter,0,map+i*DIMY+j);

            y+=four_step;
        }
        x+=xstep;
    }
}


