% Exercise 5 - Unsupervised classification - my kmeans solution
% Version: October 25, 2016
% Author(s): Frank de Morsier


% My K-means function
function data_cluster_id=my_kmeans_solution(data,k,n_iters)
    % Initial parameters:
    data_cluster_id = ones(size(data,1),1); % function lambda: init all data assigned to cluster 1

    % Initialize k centres (from randomly chosen data points)
    % HERE YOUR CODE: use randperm() function
    centres_id = randperm(size(data,1),k);
    centres = data(centres_id,:);

    % Main loop of algorithm
    for n = 1:n_iters

      % Save old centres and old sample assignment to check for termination
      old_centres = centres;
      old_cluster_id = data_cluster_id;

      % Calculate distances based on existing centres
      % HERE YOUR CODE: you can use function pdist2()
      % dist =
      dist = pdist2(data, centres);

      % Assign each sample to nearest centre
      % HERE YOUR CODE: you can use function min() and its second output which
      % gives you the index corresponding to the closest centre (name it
      % data_cluster_id)
      [min_dist, data_cluster_id] = min(dist, [], 2);

      % Update the centres based on the new assignment
      for ki = 1:k
            % HERE YOUR CODE: compute the mean of the data assigned to cluster ki
            centres(ki,:) = mean(data(data_cluster_id==ki,:));
      end

      % If centres are not changing position or sample assignments are
      % similar => stop iterations
      if sum(sum(centres-old_centres))==0 || sum(data_cluster_id-old_cluster_id) == 0 
            break;
      end
    end

    fprintf(['k-means done in number of iterations = ',int2str(n)])
end