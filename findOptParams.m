function [lambda_best, fact_best, F, U, M, C, S, result, D, Yf] = findOptParams(Y, d, lambda, ts, facts, stepsT)

min = 0;
%randIdcs = randperm(size(Y,1), 5000);
%Yf = Y(randIdcs,:);0
Yf = Y(randperm(size(Y, 1)), :);
for i = 1 : size(ts,2)
    %for j = 1: size(lambda2,2)
        for k = 1 : size(lambda,2)
            [F, U, M, C, S, result, D] = MultiverseRecTF3_matea (Yf(1:10000,:), d, lambda(k), ts(i), facts, stepsT);
            result 
            %= MAE(Yf,F)
            
            if( i~=1 && k~=1  && min < result)
                lambda_best(1) = lambda(k);
                %lambda_best(1) = lambda1(i);
                %lambda_best(2) = lambda2(j);
                fact_best = facts(i);
                min = result;
            elseif(i == 1 && k == 1)
                lambda_best(1) =  lambda(k);
                %lambda_best(2) = lambda2(j);
                fact_best = facts(i); 
                min = result;
            end
        end
    %end
end

end