function [n, m, c] = ratings4(data)

% number of users
n = max(data(:,1));

% number of movies
m = max(data(:,2));

c = zeros(2,1);

% number of different context for 2 context type 
for i = 3 : 4
    c(i-2,1) = max(data(:,i));
end

% initiate indices for sparse tensor
%index = zeros(size(data,1),4);

% initiate rating values for sparse tensor
%rating = zeros(size(data,1),1);

% define sparse tensor
%for i = 1 : size(data,1)
%    index(i,:) = data(i,1:4);
%    rating(i) = data(i,5);
%end

%Y = sptensor(index, rating);