//Example 5-32 Baseline C Code for Mandelbrot Set Map Evaluation from:https://www.intel.com/content/dam/www/public/us/en/documents/manuals/64-ia-32-architectures-optimization-manual.pdf

#define DIMX (64)
#define DIMY (64)
#define X_STEP (0.5f/DIMX) #define Y_STEP (0.4f/(DIMY/2)) int map[DIMX][DIMY];
__kernel void mandelbrot(__global float* map )
{
    int i,j;
    float x,y;
    for (i=0,x=-1.8f;i<DIMX;i++,x+=X_STEP) {
        for (j=0,y=-0.2f;j<DIMY/2;j++,y+=Y_STEP) {
            float sx,sy;
            int iter = 0; sx = x;
            sy = y;
            while (iter < 256){
                if (sx*sx + sy*sy >= 4.0f)
                    break;
                float old_sx = sx;
                sx = x + sx*sx - sy*sy;
                sy = y + 2*old_sx*sy;
                iter++;
            }
            map[i*DIMY+j] = iter;
        }
    }
}

//Example 5-33 Vectorized Mandelbrot Set Map Evaluation Using SSE4.1 Intrinsics from:https://www.intel.com/content/dam/www/public/us/en/documents/manuals/64-ia-32-architectures-optimization-manual.pdf
/*
__declspec(align(16)) float _INIT_Y_4[4] = {0,Y_STEP,2*Y_STEP,3*Y_STEP}; F32vec4 _F_STEP_Y(4*Y_STEP);
I32vec4 _I_ONE_ = _mm_set1_epi32(1);
F32vec4 _F_FOUR_(4.0f);
F32vec4 _F_TWO_(2.0f);;
void mandelbrot_simd() {
    int i,j;
    F32vec4 x,y;
    for (i = 0, x = F32vec4(-1.8f); i < DIMX; i ++, x += F32vec4(X_STEP)) {
        for (j = DIMY/2, y = F32vec4(-0.2f) +
                *(F32vec4*)_INIT_Y_4; j < DIMY; j += 4, y += _F_STEP_Y)
        { F32vec4 sx,sy;
            I32vec4 iter = _mm_setzero_si128(); int scalar_iter = 0;
            sx = x;
            sy = y;
            while (scalar_iter < 256) {
                int mask = 0;
                F32vec4 old_sx = sx;
                __m128 vmask = _mm_cmpnlt_ps(sx*sx + sy*sy,_F_FOUR_);
// if all data points in our vector are hitting the “exit” condition,
// the vectorized loop can exit
                if (_mm_test_all_ones(_mm_castps_si128(vmask)))
                    break;
            (continue)
// if non of the data points are out, we don’t need the extra code which blends the results
                if (_mm_test_all_zeros(_mm_castps_si128(vmask),
                        _mm_castps_si128(vmask))) { sx=x+sx*sx-sy*sy;
                    sy = y + _F_TWO_*old_sx*sy;
                    iter += _I_ONE_; }
                else
                {
// Blended flavour of the code, this code blends values from previous iteration with the values
// from current iteration. Only values which did not hit the “exit” condition are being stored;
// values which are already “out” are maintaining their value
                    sx = _mm_blendv_ps(x + sx*sx - sy*sy,sx,vmask);
                    sy = _mm_blendv_ps(y + _F_TWO_*old_sx*sy,sy,vmask); iter = I32vec4(_mm_blendv_epi8(iter + _I_ONE_,
                        iter,_mm_castps_si128(vmask)));
                    }
                    scalar_iter++;
            }
            _mm_storeu_si128((__m128i*)&map[i][j],iter);
        }
    }
}
*/
