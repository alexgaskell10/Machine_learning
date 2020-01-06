function U = wPCA(data)
    % Transpose date so is in correct shape
    data = data';
    
    % Centre data
    n = size(data, 2);
    X = data * (eye(n) - (1/n) * ones(n));
    
    % Step 1: compute dot product matrix
    A = X' * X;
    
    % Enforce symmetry
    A = (A + A')/2;
    
    % Step 2: perform eigenanalysis of dot product matrix and return in
    % descending order. Remove the final eig as it is zero
    [V,D] = eigs(A, size(A,1)-1);
    
    % Step 3: compute eigenvectors (using Lemma 1)
    U = X * V * D^-0.5;
    
    % Step 4: whiten data
    U = U * D^-0.5;
    
end