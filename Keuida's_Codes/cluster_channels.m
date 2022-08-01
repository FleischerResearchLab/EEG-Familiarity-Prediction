function [output_mat] = cluster_channels(input_mat, cluster_index)

[n_ch n_samples n_trials] = size(input_mat);

n_clusters = length(cluster_index);

output_mat = zeros(n_clusters, n_samples, n_trials);

for cc = 1:n_clusters
    temp_val = mean(input_mat(cluster_index{cc},:,:),1);
    output_mat(cc,:,:) = temp_val;
end

    