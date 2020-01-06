load fea;
% load gnd;
load idx;

id = idx;
test_idx = 1:size(fea, 1);

data = fea;
n = size(data, 2);
X = data * (eye(n) - (1/n) * ones(n) * ones(n)');
size(X)
% % g = size(gnd);
% % i = size(idx);
% 
% A = X' * X;
% 
% A = (A + A')/2;
% 
% % Step 2: perform eigenanalysis of dot product matrix
% % [V,D] = eigs(A, size(A,1)-1);
% e = eigs(A, size(A,1)-1);
% 
% size(e)
% e(1:20)
% % D(90:100,90:100)
% % size(D)
% % D(end:-1:end-10,end:-1:end-10)
% % U = X * D * V^-0.5;
% % size(U)



