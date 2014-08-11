
#include "mex.h"
#include "cuda_ops.h"


void mexFunction( int nout, mxArray *pout[], int nin, const mxArray *pin[])
{
    // Get the pointers to the data. 
    // All of the memory allocation stuff is done in 
    // cublas_matrix_multiply including the imaginary
    // part checking.
    if (!mxIsSingle(pin[0]) || !mxIsSingle(pin[1])){
        mexErrMsgIdAndTxt( "MATLAB:matrix_mult:inputerror",
                "Both input matrices must be single precision.");
    }
    
    float *h_A = (float*)mxGetData(pin[0]);
    float *h_A_im = (float*)mxGetImagData(pin[0]);
    float *h_B = (float*)mxGetData(pin[1]);
    float *h_B_im = (float*)mxGetImagData(pin[1]);
    
    unsigned int A_m = mxGetM(pin[0]);
    unsigned int A_n = mxGetN(pin[0]);
    
    unsigned int B_m = mxGetM(pin[1]);
    unsigned int B_n = mxGetN(pin[1]);
    
    unsigned int C_m = A_m;
    unsigned int C_n = B_n;
    
    // TODO: probably should check to make sure the input 
    // is single precision and error / convert to single
    // if possible.
    if (A_n != B_m){
        mexErrMsgIdAndTxt( "MATLAB:matrix_multiply:inputerror",
                "Inner matrix dimensions must agree");
    }
    
    // set up outputs 
    pout[0] = mxCreateNumericMatrix((mwSize) C_m, (mwSize) C_n,
            mxSINGLE_CLASS, mxCOMPLEX);
    float *h_C = (float*)mxGetData(pout[0]);
    float *h_C_im = (float*)mxGetImagData(pout[0]);

    cublas_matrix_multiply(h_A, h_A_im, h_B, h_B_im, h_C, h_C_im, A_m, A_n, B_m, B_n, C_m, C_n);
}
