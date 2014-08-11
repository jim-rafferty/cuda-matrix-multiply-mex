#include "cuda_ops.h"
#include <cublas_v2.h>

void cublas_matrix_multiply(float *A, float *ImA, float *B, float *ImB, float *C, float *ImC, 
        unsigned int A_m, unsigned int A_n, 
        unsigned int B_m, unsigned int B_n, 
        unsigned int C_m, unsigned int C_n){

    using namespace std;
    // NB: matlab stores complex numbers as separate real and immaginary parts
    // to use the cublas lib we must convert matlabs 2 floats to cuComplex.
    // (which is the same as float2)
    // def complex variables.
    cuComplex *mat_A = new cuComplex[A_m * A_n];
    cuComplex *mat_B = new cuComplex[B_m * B_n];
    cuComplex *mat_C = new cuComplex[C_m * C_n];
    // copy floats to cuComplex
	int i;
	for (i = 0; i < A_m * A_n; i++){;
		mat_A[i].x = A[i]; // real part
        // Im part
        // If the Im part is not present, set it to 0
        // A real matrix will use twice as much memory as nessecary,
        // but it's likely that the input was a double from matlab
        // anyway...
        if (ImA == NULL){
            mat_A[i].y = 0;
        }
        else{
    		mat_A[i].y = ImA[i]; 
        }
	} 
    // repeat operation for matrix B
	for (i = 0; i < B_m * B_n; i++){;
		mat_B[i].x = B[i];
        if (ImB == NULL){
    		mat_B[i].y = 0;
        }
        else{
            mat_B[i].y = ImB[i];
        }
	}

    // def GPU variables
    cuComplex *nv_A;
    cuComplex *nv_B;
    cuComplex *nv_C;

    // allocate mem for GPU vars
    cudaMalloc((void **) &nv_A, A_m * A_n * sizeof(cuComplex));
    cudaMalloc((void **) &nv_B, B_m * B_n * sizeof(cuComplex));
    cudaMalloc((void **) &nv_C, C_m * C_n * sizeof(cuComplex));

    // copy data to GPU
    cudaMemcpy(nv_A, mat_A, A_m * A_n * sizeof(cuComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(nv_B, mat_B, B_m * B_n * sizeof(cuComplex), cudaMemcpyHostToDevice);

    cuComplex alf; alf.x = 1; alf.y = 0; 
    cuComplex bet; bet.x = 0; bet.y = 0;
    const cuComplex *alpha = &alf;
    const cuComplex *beta = &bet;
    // Create a handle for CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Do the actual multiplication
    cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, A_m, B_n, A_n, alpha, nv_A, A_m, nv_B, A_n, beta, nv_C, A_m);

    // Destroy the handle
    cublasDestroy(handle);

    // copy solution back
    cudaMemcpy(mat_C, nv_C, C_m * C_n * sizeof(cuComplex), cudaMemcpyDeviceToHost);

    // copy complex float to separate floats.
	for (i = 0; i < C_m * C_n; i++){;
        C[i] = mat_C[i].x;
        ImC[i] = mat_C[i].y;
	}

    // clean up GPU vars
    cudaFree(nv_A);
    cudaFree(nv_B);
    cudaFree(nv_C);

    // clean up complex vars
    free(mat_A);
    free(mat_B);
    free(mat_C);
}

