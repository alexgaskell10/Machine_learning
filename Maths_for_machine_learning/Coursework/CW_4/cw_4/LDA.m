function U = LDA(data, labels)
    % Transpose date so is in correct shape
    X = data';
    y = labels';
    
    % Compute centering matrix M
    n = size(X, 2);
    c = length(unique(y));
    M = zeros(n);
    col_id = 1;
    for i = 1:c
        % Find of class i
        ci = sum(y(:) == i);
        % Form mean matrix for class i
        E = (1/ci) * ones(ci);
        % Add mean matrix for class i to centering matrix M
        M(col_id : col_id + ci - 1, col_id : col_id + ci - 1) = E;
        % Increment counter for next iteration
        col_id = col_id + ci;
    end
    
    % Compute Xw
    Xw = X * (eye(n) - M);
    
    % Perform eigenanalysis on Sw = Xw' Xw
    Sw = Xw' * Xw;
    Sw = (Sw + Sw') / 2;
    [V,D] = eigs(Sw, n - (c + 1));   % Only keep n - (c + 1) non-zero eigenvalues
        
    % Perform whitening transformation on sw
    U = Xw * V * D^-1;
    
    % Project data to get Xb
    Xb = U' * X * M;
    
    % Perform PCA on Xb
    A = Xb * Xb';
    A = (A + A') / 2;
    [Q,D] = eigs(A, c - 1); % Error here- only keep 67 features (c-1)
        
    % Compute total transform W = UQ
    U = U * Q;
    
end