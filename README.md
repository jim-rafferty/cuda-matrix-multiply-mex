cuda-matrix-multiply-mex
========================

A mex function to perform matrix multiplication on an nvidia gpu with a potentially huge improvement in performance depending on hardware available. Matlab's parallel computing toolbox is not required.


This works by separately compiling a cuda function doing the matrix multiplication and a mex function reading the data innput from matlab into objects and then linking them together. Matlab knows nothing about cuda and vice versa.


The included compile_matrix_multiply.m matlab function will require minimal editing to get it working under a generic Linux installation, and probably a mac too tough I've not tested this. The trick should in principle work under windows but I've not tried to get it to work. I'd be most grateful to be able to merge in a pull request on this!


Files:

matrix_multiply.cpp - c++ source file for the mex function. Doesn't know about cuda other than the inclusion of the local header cuda_ops.h

cuda_ops.* - cuda header and implementation of the cuda code that does the matrix multiplication. 

compile_matrix_multiply.m - matlab function to compile under linux. Will probably need a bit of editing to get working. I've initialised the bin, include and lib paths with the default for a 64 bit os. 

test_cuda.m - a function that does some matrix multiplications in matlab and with the compiled function, testing the difference between the results and time taken to process. 
