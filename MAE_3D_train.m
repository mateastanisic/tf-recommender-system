%% MEAN ABSOLUTE ERROR
%
%  F represents data with predicted ratings for the same data that we have
%  real rating values in 'data'. Meaning, F(:,1:3) == data(:,1:3)!!!
%
%  This is used because it is much more faster than original version.
%
%  Return value is calculated mean absolute error with formula:
%  1/K * sum{i,j,k -> n,m,c}(Dijk * || Fijk - Yijk||)
%
function result = MAE_3D_train(F, data)

% Number of ratings. (Number of non-zeros in tensor.)
%K = nnz(Y);
K = size(data,1);

Fr = F(:,4);
Yr = data(:,6);

suma = sum(abs(Fr-Yr));


result = suma / K;

end