function [dist] = deck_distance(deck, Mu, type)
%DECK_DISTANCE Calculate the distance between a partially filled deck and
%the centroids
%
%   input -----------------------------------------------------------------
%   
%       o deck : a partially filled deck
%       o Mu : Value of the centroids
%       o type : type of distance to use {'L1', 'L2', 'Linf'}
%
%   output ----------------------------------------------------------------
%
%       o dist : k X M the distance to the k centroids
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

idx = find(deck);
cards = nonzeros(deck);
Mu_corr = zeros(length(idx), size(Mu,2));

for i=1:length(idx)
    Mu_corr(i,:) = Mu(idx(i),:);
end

dist = distance_to_centroids(cards, Mu_corr, type);
    
    
end

