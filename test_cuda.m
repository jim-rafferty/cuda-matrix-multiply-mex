function iterated_tests()
s = tic;

nIter = 10;

fprintf('\nSquare matrix tests\n\n');

cuda_t_list = [];
matlab_t_list = [];
difference_list = [];

for kIter = 1:nIter
    ns_dim1 = randi([1000, 2000]);
    A = rand(ns_dim1) + i * rand(ns_dim1);
    B = rand(ns_dim1) + i * rand(ns_dim1);
    
	tic;
    c_cuda = matrix_multiply(single(A), single(B));
    t_cuda = toc;
    tic;
    c_matlab = A * B;
    t_matlab = toc;
    
    diffs = abs(c_cuda - c_matlab);
    fprintf('Final size %ux%u. mean abs diff = %f. Time: matlab: %.2fs cuda: %.2fs\n',...
        ns_dim1, ns_dim1, mean(diffs(:)), t_matlab, t_cuda);
    
    cuda_t_list = cat(1, cuda_t_list, t_cuda);
    matlab_t_list = cat(1, matlab_t_list, t_matlab);
    difference_list = cat(1, difference_list, mean(diffs(:)));
end

fprintf('\nNon square matrix tests\n\n');
for kIter = 1:nIter
    ns_dim1 = randi([1000, 2000]);
    ns_dim2 = randi([1000, 2000]);
    ns_dim3 = randi([1000, 2000]);
    
    A = rand(ns_dim1, ns_dim2) + i * rand(ns_dim1, ns_dim2);
    B = rand(ns_dim2, ns_dim3) + i * rand(ns_dim2, ns_dim3);
    
	tic;
    c_cuda = matrix_multiply(single(A), single(B));
    t_cuda = toc;
    tic;
    c_matlab = A * B;
    t_matlab = toc;
    
    diffs = abs(c_cuda - c_matlab);
    fprintf('Final size %ux%u. mean abs diff = %f. Time: matlab: %.2fs cuda: %.2fs\n',...
        ns_dim1, ns_dim3, mean(diffs(:)), t_matlab, t_cuda);
    
    cuda_t_list = cat(1, cuda_t_list, t_cuda);
    matlab_t_list = cat(1, matlab_t_list, t_matlab);
    difference_list = cat(1, difference_list, mean(diffs(:)));    
end
fprintf('\n*******\n');
fprintf('Summary\n');
fprintf('*******\n\n');

fprintf('Average absolute difference: %f\n', mean(difference_list));
fprintf('Total cuda time: %.2fs\n', sum(cuda_t_list));
fprintf('Total matlab time: %.2fs\n', sum(matlab_t_list));
s = toc(s);
fprintf('Total elapsed time: %.2fs\n',s);
