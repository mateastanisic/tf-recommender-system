%% Helper function for tensor multiplication
function [best_p, best_i] = recommend_this(S, U, M ,C, user, context)
    result = [];
    movies = 106362; %number of movies
    
    for i = 1 : movies
        predict = tensorMult(tensor(S.data), U(user,:), M(i,:), C(context,:));
        result = [result, predict];
    end
    
    [best_p, best_i] =  maxk(result,100);
end