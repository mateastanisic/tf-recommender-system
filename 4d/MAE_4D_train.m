%% MEAN ABSOLUTE ERROR for 4D tensor
%
function result = MAE_4D_train(F, data)

% Number of ratings
K = size(data,1);

Fr = F(:,5);
Yr = data(:,5);

suma = sum(abs(Fr-Yr));

result = suma / K;
end