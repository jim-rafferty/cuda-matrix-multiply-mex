function compile_matrix_multiply()

% Edit these lines to reflect your cuda install location
cuda_inc = '/usr/local/cuda/include';
cuda_lib = '/usr/local/cuda/lib64';
nvcc_bin = '/usr/local/cuda/bin/nvcc';

% if compilation fails with gcc version not supported error, try adding
% something like the following to NVFLAGS to point to a supported gcc version: 
% -ccbin /usr/bin/g++-4.8
NVFLAGS = sprintf('-I%s -c -Xcompiler -fpic', cuda_inc);
MXFLAGS = '-c -largeArrayDims';
MX_LINK_FLAGS = sprintf('-largeArrayDims -cxx -L%s -lcudart -lcublas', ...
    cuda_lib);

% compile cuda
fprintf('Running:%s cuda_ops.cu %s\n', nvcc_bin, NVFLAGS)
system(sprintf('%s cuda_ops.cu %s\n', nvcc_bin, NVFLAGS));

% compile mex
fprintf('Running:mex matrix_multiply.cpp %s\n', MXFLAGS)
eval(sprintf('mex matrix_multiply.cpp %s', MXFLAGS));

% link
fprintf('Running:mex matrix_multiply.o cuda_ops.o %s', MX_LINK_FLAGS)
eval(sprintf('mex matrix_multiply.o cuda_ops.o %s', MX_LINK_FLAGS));