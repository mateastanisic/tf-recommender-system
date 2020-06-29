
%% Helper function for creating sparse tensor Y and D
function [Y, n, m, c] = ratings(data)

% number of users
n = max(data(:,1));

% number of movies
m = max(data(:,2));

% number of different context for one context type ( this is for one
% context only )
c = max(data(:,3));

% initiate indices for sparse tensor
index = zeros(size(data,1),3);

% initiate rating values for sparse tensor
rating = zeros(size(data,1),1);

% define sparse tensor
for i = 1 : size(data,1)
    index(i,:) = data(i,1:3);
    rating(i) = data(i,4);
end

Y = sptensor(index, rating);
D = sptensor(index, ones(size(data,1),1));

end