function SVM()

    % Solve the quadratic optimisation problem. Estimate the labels for 
    % each of the test samples and report the accuracy of your trained SVM 
    % utilising the ground truth labels for the test data.

    load('X.mat');          % X = f x n = 100 X 2000
    load('l.mat');
    load('X_test.mat');
    load('l_test.mat');
    
    % Init variables
    [f, n] = size(X);
    S_t = (1 / n) * (X * X');
    C = 1;                % <-- this is penalty parameter C
    threshold = 10^-6;

    % Form quadprog inputs ( (y y') elem_mult (X' inv(S_t) X) )
    H = (l * l') .* (X' * inv(S_t) * X);
    H = (H + H') / 2;
    A = zeros(n);
    f = -ones(1, n);
    c = zeros(n, 1);  
    c_e = 0;
    g_l = zeros(n, 1);
    g_u = C * ones(n, 1);    % Inf(n, 1);
    A_e = l';
    
    % Run quadprog
    alpha = quadprog(H, f, A, c, A_e, c_e, g_l, g_u);

    % enforce alpha_i -> 0 below threshold. These are the support vectors.
    non_support_ids = find(alpha < threshold);
    alpha(non_support_ids) = 0;

    % Compute w = inv(S_t) sum (a_i y_i x_i)
    w = inv(S_t) * X * (l .* alpha);
    
    % Compute b by averaging over all support vectors
    SV_ids = find(alpha > 0);
    b = (1/size(SV_ids, 1)) * sum(l(SV_ids)' - w' * X(:, SV_ids));

    % Compute Xi = max(0, 1 - y_i(w' X + b))
    xi = max(0, 1 - l .* (X' * w + b));

    % Compute predictions
    l_pred = X_test' * w + b;
    l_pred(l_pred < 0) = -1;
    l_pred(l_pred > 0) = 1;

    accuracy = length(find((l_pred - l_test) == 0)) / length(l_pred);
    fprintf('Accuracy on the test set is %3.2f\n', accuracy);

end
