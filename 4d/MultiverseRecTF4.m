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
function [F, U, M, C, S, cv_errors, cv_test_error, cvErr, testErr ] = MultiverseRecTF4(data_train, data_test, data_size, d, alpha, beta, steps, cv)

% shuffle random
rng('shuffle');

% shuffle data
data = data_train(randperm(size(data_train(1:data_size,:), 1)), :);


% matrix factorization params
% for yahoo, params: alfa 1*10-3 i beta 0.1
%alpha = 0.001% 0.0002;
%beta = 0.1 % 0.02;

disp("starting cv")

load('fisheriris');

CVO = cvpartition(data_size,'k',cv);
cv_errors = zeros(CVO.NumTestSets,1);
cv_test_error = zeros(CVO.NumTestSets,1);


for cv_iter = 1 : CVO.NumTestSets
    disp("");
    disp("");
    fprintf('CV: %i\n', cv_iter);
    
    % Make sparse tensor Y with ratings. 
    % We have to make Y from original train data, not smaller because we
    % may not include all users/items in choosen subset of train data and
    % therefore we will not be able to test whole testset.
    [n, m, c] = ratings4(data_train);
    disp("done ratings()");
    disp(n);
    disp(m);
    disp(c);

    % Initialization of U, M, C, S with small values.
    [U, M, C, S] = initialization(d, n, m, c, 1);
    disp("done initialization");
    
    % define train and test
    trainIdx = CVO.training(cv_iter);
    testIdx = CVO.test(cv_iter);
    
    disp("starting tf");
    % iteration over cells with raitings

    train = define_data(data, trainIdx);%define train
    test = define_data(data, testIdx); %define test
    
    % Only cells with indices same as in train set will be important 
    % in process of making a model.
    %F = sptensor(indices, zeros(size(original_data,1),1));
    F = zeros(size(train));
    
    for s = 1 : steps
        disp("");
        fprintf('STEP: %i\n', s);

        % start timer
        tic
        
        for iter = 1 : size(train,1)
            if(mod(iter,1000)==0)
                disp("")
                disp(iter)
                toc
            end
            
            i = train(iter,1);
            j = train(iter,2);
            k = train(iter,3);
            l = train(iter,4);

            % update only if relevant for mae
            F(iter,:) = [i,j,k,l,tensorMult(S, U(i,:), M(j,:), C{1}(k,:), C{2}(l,:))];

            % save old values
            Ui = U(i,:);
            Mj = M(j,:);
            Ck1 = C{1}(k,:);
            Ck2 = C{2}(l,:);

            % update 
            [ U(i, :), M(j, :), C{1}(k, :), C{2}(l, :), S ] = update(alpha, beta, Ui, Mj, Ck1, Ck2, F(iter,5), train(iter,5), S);

        end

        % check if we found good aproximation of Y
        step_error = MAE_4D_train(F, train)
  
        if step_error < 0.1
            filename = strcat("saved/amazon_4d_cv5_", string(cv_iter) ,"_150000_size_20_20_5_3_step", string(s),".mat");
            save(filename)
            break
        end

        fprintf('STEP: %i  DONE!\n', s);
        % end timer
        toc

        %save workspace
        filename = strcat("saved/amazon_4d_cv5_", string(cv_iter) ,"_150000_size_20_20_5_3_step", string(s),".mat");
        save(filename)
    end
    
    % Calculate test errors for this cv test set and original test set.
    cv_errors(cv_iter) = MAE_4D_test(U, M, C, S, test)
    cv_test_error(cv_iter) = MAE_4D_test(U, M, C, S, data_test)
    
    % print time
    toc
    
end
cvErr = sum(cv_errors)/sum(CVO.TestSize);
testErr = sum(cv_test_error)/sum(CVO.TestSize);

end 


%% initialization
function [U, M, C, S] = initialization (d, n, m, c, fact)

% initiale with small numbers ( 1 - 5 ?)
U = randi(fact, n, d(1)) .* rand(n, d(1));
M = randi(fact, m, d(2)) .* rand(m, d(2));
for i = 1:size(c,1)
    C{i} = randi(fact, c(i,1), d(i+2)) .* rand(c(i,1), d(i+2));
end
S = randi(fact, d(1), d(2), d(3), d(4)) .* tensor(rand(d(1), d(2), d(3), d(4)));

end

%% tensor multiplication
function result = tensorMult(S, Ui, Mj ,Ck, Cl)

    %tensor matrix multiplication
    %F_{i,j,k} = S x_U Ui X_M Mj X_C Ck
    
    Y = ttm(S,Ui,1);

    Y2 = ttm(Y,Mj,2);

    Y3 = ttm(Y2,Ck,3);
    
    Y4 = ttm(Y3,Cl,4);
    
    result = squeeze(Y4);

end

%% least squared loss update 
function [Unew, Mnew, C1new, C2new, Snew] = update(alpha, beta, Ui, Mj, Ck, Cl, Fval, Yval, S)

    %% update U
    loss = sign(Fval - Yval) * S;
    
    R1 = ttm(loss, Mj, 2);
    R2 = ttm(R1, Ck, 3);
    R3 = ttm(R2, Cl, 4);

    Unew = Ui - alpha * ( beta * Ui + double(R3)' );

    %% update M
    R1 = ttm(loss, Ui, 1);
    R2 = ttm(R1, Ck, 3);
    R3 = ttm(R2, Cl, 4);

    Mnew = Mj - alpha * ( beta * Mj + double(squeeze(R3))' );
    
    %% update C1
    R1 = ttm(loss, Ui, 1);
    R2 = ttm(R1, Mj, 2);
    R3 = ttm(R2, Cl, 4);

    C1new = Ck - alpha * ( beta * Ck + double(squeeze(R3))' );
    
    %% update C2
    R1 = ttm(loss, Ui, 1);
    R2 = ttm(R1, Mj, 2);
    R3 = ttm(R2, Ck, 3);
    
    C2new = Cl - alpha * ( beta * Cl + double(squeeze(R3))' );
  
    
    %% update S
    loss = sign(Fval - Yval) * Ui;
    
    R1 = ttt(tensor(loss), tensor(Mj));
    R2 = ttt(tensor(R1),   tensor(Ck));
    R3 = ttt(tensor(R2),   tensor(Cl));

    Snew =  tensor( S - alpha * beta * S ) - tensor( alpha * squeeze(R3) );
    
end

%% Helper for defining train and test data
function data_partition = define_data(original_data, indices)

    data_partition = zeros(sum(indices), size(original_data,2) );
    index = 1;

    for i = 1 : size(indices,1)
        if indices(i)
            data_partition(index,:) = original_data(i,:);
            index = index + 1;
        end
    end

end

