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
function result = MAE_4D_test(U, M, C, S, data)

% Number of ratings. 
K = size(data,1);

% Initialize suma with zero.
suma = 0;

for iter = 1 : size(data,1)
    i = data(iter,1);
    j = data(iter,2);
    k = data(iter,3);
    l = data(iter,4);
    
    predict = tensorMult(S, U(i,:), M(j,:), C{1}(k,:), C{2}(l,:));
    
    suma = suma + abs( predict - data(iter,5) );
end

result = suma / K;

end