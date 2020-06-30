%% Helper function for tensor multiplication
function result = tensorMult(S, Ui, Mj ,Ck)

    %tensor matrix multiplication
    %F_{i,j,k} = S x_U Ui X_M Mj X_C Ck
    
    Y = ttm(S,Ui,1);

    Y2 = ttm(Y,Mj,2);

    Y3 = ttm(Y2,Ck,3);
    
    result = squeeze(Y3);

end