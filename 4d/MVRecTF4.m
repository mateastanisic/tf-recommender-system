function [F, U, M, C, S, result, error] = MVRecTF7(Train, Test, d, lambda, ts, facts, stepsT)

%train part

    %Yf = Y(randperm(size(Y, 1)), :);

    [F, U, M, C, S, result, ~] = MultiverseRecTF7 (Train, d, lambda, ts, facts, stepsT);
    result
    
    error = test(Test,S,U,M,C)

end


function error = test(data, S, U, M, C)

[Y, ~, ~, ~, D, indices] = ratings7(data);
F = sptensor(indices, zeros(size(data,1),1));
for iter = 1 : size(data,1)
    
    i = indices(iter,1);
    j = indices(iter,2);
    k = indices(iter,3);
    l = indices(iter,4);
    m = indices(iter,5);
    n = indices(iter,6);
    o = indices(iter,7);
    if j<=11912
        
        F(i,j,k, l, m, n, o) = tensorMult(S, U(i,:), M(j,:), C{1}(k,:), C{2}(l,:), C{3}(m,:), C{4}(n,:), C{5}(o,:));
    end
    
end
error = MAE7(D, Y, F, data);
end

function result = tensorMult(S, Ui, Mj ,Ck, Cl, Cm, Cn, Co)

    %tensor matrix multiplication
    %F_{i,j,k} = S x_U Ui X_M Mj X_C Ck
    
    Y = ttm(S,Ui,1);

    Y2 = ttm(Y,Mj,2);

    Y3 = ttm(Y2,Ck,3);
    
    Y4 = ttm(Y3,Cl,4);
    
    Y5 = ttm(Y4,Cm,5);
    
    Y6 = ttm(Y5,Cn,6);
    
    Y7 = ttm(Y6,Co,7);
    
    result = squeeze(Y7);

end