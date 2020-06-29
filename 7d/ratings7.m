function [Y, n, m, c, D, index] = ratings7(data)

% number of users
n = max(data(:,1));

% number of movies
m = max(data(:,2));

c = zeros(size(data,2)-3,1);

% number of different context for five context type 
for i = 3: size(data,2)-1 
    c(i-2,1) = max(data(:,i));
end
% initiate indices for sparse tensor
index = zeros(size(data,1),size(data,2)-1);

% initiate rating values for sparse tensor
rating = zeros(size(data,1),1);

% define sparse tensor
for i = 1 : size(data,1)
    index(i,:) = data(i,1:size(data,2)-1 );
    rating(i) = data(i,size(data,2));
end

Y = sptensor(index, rating);
D = sptensor(index, ones(size(data,1),1));