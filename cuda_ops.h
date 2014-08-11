#ifndef CU_MEX_H
#define CU_MEX_H

void cublas_matrix_multiply(float *A, float *ImA, float *B, float *ImB, float *C, float *ImC, 
        unsigned int A_m, unsigned int A_n, 
        unsigned int B_m, unsigned int B_n, 
        unsigned int C_m, unsigned int C_n);

#endif
