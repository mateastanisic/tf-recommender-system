%% MEAN ABSOLUTE ERROR
%
%  D is (sparse) tensor with binary values representing if we have defined
%  rating of user i for item j in context k
%
%  F is tensor with calculated ratings
%
%  Y is (sparse) tensor with original reatings 
%
%  Return value is calculated mean absolute error with formula:
%  1/K * sum{i,j,k -> n,m,c}(Dijk * || Fijk - Yijk||)
%
function result = MAE7(D, Y, F, data)


%resultg = collapse(abs(Y-F),1:3)


% Initialize sum to zero.
suma = 0;

% Number of ratings. (Number of non-zeros in tensor.)
K = nnz(Y);

for iter = 1 : K
    i = data(iter,1);
    j = data(iter,2);
    k = data(iter,3);
    l = data(iter,4);
    m = data(iter,5);
    n = data(iter,6);
    o = data(iter,7);
        
    suma = suma + abs(F(i, j, k, l, m, n, o) - Y(i, j, k, l, m, n, o) );
end

result = suma / K;

end