%% MULTIVERSE RECOMENDATION WITH TF; VERSION WITH 3D TENSOR
%  
%  Variable 'data' represents dataset with user ratings.
%  Variable 'd' is arary with specified values for d_U, d_M and d_C.
%  Variable 'lambda' is array with specified values for lambda_U, lambda_M,
%  and lambda_C needed for regularization.
%  Variable 'fact' is factor for initialiating U, M, C and S.
%
%  Function returns variables:
%       F - aproximation of sparse tensor Y with additional, 
%       previously unknown, ratings
%       U - matrix with dimensions n x d_U 
%       (every row represents the strength of the associations between a
%       user and the features)?
%       M - matrix with dimensions m x d_M
%       (every row represents the strength of the associations between a
%       movie and the features)?
%       C - matrix with dimensions c x d_C
%       (every row represents the strength of the associations between a
%       context and the features)?
%       S - central tensor with dimensions d_U x d_M x d_C
%
%
function [F, U, M, C, S, error, D] = MultiverseRecTF7(data, d, lambda, t0, fact, stepsT)
% Y, d, t_0, lambda, n_users, n_movies, n_con, fact
% n_users, n_movies, n_con = broj korisnika, filmova, konteksta
% fact = mali broj koji se koristi za inicijalizaciju

% shuffle random
rng('shuffle');

% make sparse tensor Y with ratings and sparse binary tensor D representing
% if user has rated movie in some context
[Y, n, m, c, D, indices] = ratings7(data);
disp("done ratings()");
n
m

% initialization of U, M, C, S
[U, M, C, S] = initialization(d, n, m, c, fact);
disp("done initialization");

% only cells with indices same as in Y are relevant fro model training
F = sptensor(indices, zeros(size(data,1),1));

% matrix factorization params
alpha = t0% 0.0002;
beta =lambda % 0.02;

disp("starting tf");
% iteration over cells with raitings
steps = stepsT;
for s = 1 : steps
    disp("");
    fprintf('STEP: %i\n', s);
    
    % start timer
    tic
    for iter = 1 : size(data,1)
        if(mod(iter,1000)==0)
            disp("")
            disp(iter)
            toc
        end
        
        index = zeros(size(indices,2));
        
        %size(indices,2)
        
        for t = 1:size(indices,2)
            index(t) = indices(iter,t);
        end
        %i = indices(iter,1);
        %j = indices(iter,2);
        %k = indices(iter,3);
        
        % update only if relevant for mae
        F(index(1),index(2),index(3),index(4),index(5),index(6),index(7)) = tensorMult(S, U(index(1),:), M(index(2),:), C{1}(index(3),:), C{2}(index(4),:), C{3}(index(5),:), C{4}(index(6),:), C{5}(index(7),:));

        % save old values
        Ui = U(index(1),:);
        Mj = M(index(2),:);
        Ck1 = C{1}(index(3),:);
        Ck2 = C{2}(index(4),:);
        Ck3 = C{3}(index(5),:);
        Ck4 = C{4}(index(6),:);
        Ck5 = C{5}(index(7),:);

        % update 
        [ U(index(1), :), M(index(2), :), C{1}(index(3), :), C{2}(index(3), :), C{3}(index(4), :), C{4}(index(6), :), C{5}(index(7), :), S ] = update(alpha, beta, Ui, Mj, Ck1, Ck2, Ck3, Ck4, Ck5,  F(index(1),index(2),index(3),index(4),index(5),index(6),index(7)),  Y(index(1),index(2),index(3),index(4),index(5),index(6),index(7)), S);

    end
    
    % check if we found good aproximation of Y
    error = MAE7(D, Y, F, data)
    if error < 0.001
        break
    end
    
    % end timer
    toc
end


end

%% initialization
function [U, M, C, S] = initialization (d, n, m, c, fact)

% initiale with small numbers ( 1 - 5 ?)
U = randi(fact, n, d(1)) .* rand(n, d(1));
M = randi(fact, m, d(2)) .* rand(m, d(2));
for i = 1:size(c,1)
    C{i} = randi(fact, c(i,1), d(i+2)) .* rand(c(i,1), d(i+2));
end
S = randi(fact, d(1), d(2), d(3), d(4), d(5), d(6), d(7)) .* tensor(rand(d(1), d(2), d(3), d(4), d(5), d(6), d(7)));

end

%% tensor multiplication
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

%% least squared loss update 
function [Unew, Mnew, C1new, C2new, C3new, C4new, C5new, Snew] = update(alpha, beta, Ui, Mj, Ck, Cl, Cm, Cn, Co, Fval, Yval, S)

    %% update U
    loss = sign(Fval - Yval) * S;
    
    R1 = ttm(loss, Mj, 2);
    R2 = ttm(R1, Ck, 3);
    R3 = ttm(R2, Cl, 4);
    R4 = ttm(R3, Cm, 5);
    R5 = ttm(R4, Cn, 6);
    R6 = ttm(R5, Co, 7);

    Unew = Ui - alpha * ( beta * Ui + double(R6)' );

    %% update M
    R1 = ttm(loss, Ui, 1);
    R2 = ttm(R1, Ck, 3);
    R3 = ttm(R2, Cl, 4);
    R4 = ttm(R3, Cm, 5);
    R5 = ttm(R4, Cn, 6);
    R6 = ttm(R5, Co, 7);

    Mnew = Mj - alpha * ( beta * Mj + double(squeeze(R6))' );
    
    %% update C1
    R1 = ttm(loss, Ui, 1);
    R2 = ttm(R1, Mj, 2);
    R3 = ttm(R2, Cl, 4);
    R4 = ttm(R3, Cm, 5);
    R5 = ttm(R4, Cn, 6);
    R6 = ttm(R5, Co, 7);

    C1new = Ck - alpha * ( beta * Ck + double(squeeze(R6))' );
    
    %% update C2
    R1 = ttm(loss, Ui, 1);
    R2 = ttm(R1, Mj, 2);
    R3 = ttm(R2, Ck, 3);
    R4 = ttm(R3, Cm, 5);
    R5 = ttm(R4, Cn, 6);
    R6 = ttm(R5, Co, 7);

    C2new = Cl - alpha * ( beta * Cl + double(squeeze(R6))' );
    
    %% update C3
    R1 = ttm(loss, Ui, 1);
    R2 = ttm(R1, Mj, 2);
    R3 = ttm(R2, Ck, 3);
    R4 = ttm(R3, Cl, 4);
    R5 = ttm(R4, Cn, 6);
    R6 = ttm(R5, Co, 7);

    C3new = Cm - alpha * ( beta * Cm + double(squeeze(R6))' );
    
    %% update C4
    R1 = ttm(loss, Ui, 1);
    R2 = ttm(R1, Mj, 2);
    R3 = ttm(R2, Ck, 3);
    R4 = ttm(R3, Cl, 4);
    R5 = ttm(R4, Cm, 5);
    R6 = ttm(R5, Co, 7);

    C4new = Cn - alpha * ( beta * Cn + double(squeeze(R6))' );
    
    %% update C5
    R1 = ttm(loss, Ui, 1);
    R2 = ttm(R1, Mj, 2);
    R3 = ttm(R2, Ck, 3);
    R4 = ttm(R3, Cl, 4);
    R5 = ttm(R4, Cm, 5);
    R6 = ttm(R5, Cn, 6);

    C5new = Co - alpha * ( beta * Co + double(squeeze(R6))' );
    
    %% update S
    loss = sign(Fval - Yval) * Ui;
    
    R1 = ttt(tensor(loss), tensor(Mj));
    R2 = ttt(tensor(R1),   tensor(Ck));
    R3 = ttt(tensor(R2),   tensor(Cl));
    R4 = ttt(tensor(R3),   tensor(Cm));
    R5 = ttt(tensor(R4),   tensor(Cn));
    R6 = ttt(tensor(R5),   tensor(Co));

    Snew =  tensor( S - alpha * beta * S ) - tensor( alpha * squeeze(R6) );
    
end

