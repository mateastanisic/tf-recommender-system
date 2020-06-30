%% MEAN ABSOLUTE ERROR
%
%  This version of calculating MAE is when 'data' represents ratings not
%  previously known. Meaning, 'data' has some test rows/is testset.
%  That is why we need U, M, C and S - they represent trained model and
%  we will use it for making a rating prediction for some 
%  (user,item,context) tiplet.
%
%  Return value is calculated mean absolute error with formula:
%  1/K * sum{i,j,k -> n,m,c}(Dijk * || Fijk - Yijk||)
%
function result = MAE_3D_test(U, M, C, S, data)

% Number of ratings. 
K = size(data,1);

% Initialize suma with zero.
suma = 0;

for iter = 1 : size(data,1)
    i = data(iter,1);
    j = data(iter,2);
    k = data(iter,3);
    
    predict = tensorMult(S, U(i,:), M(j,:), C(k,:));
    
    if predict > 5
        predict = 5;
    end
    if predict < 1
        predict = 1;
    end
    
    suma = suma + abs( predict - data(iter,6) );
end

result = suma / K;

end

%% Helper function for tensor multiplication
function result = tensorMult(S, Ui, Mj ,Ck)

    %tensor matrix multiplication
    %F_{i,j,k} = S x_U Ui X_M Mj X_C Ck
    
    Y = ttm(S,Ui,1);

    Y2 = ttm(Y,Mj,2);

    Y3 = ttm(Y2,Ck,3);
    
    result = squeeze(Y3);

end