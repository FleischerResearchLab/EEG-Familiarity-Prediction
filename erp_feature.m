% compute ERP features
% ERP features are time avg of signal over window size
% dim of feature is characterized by win_size
% computed for each channel
%
% INPUTS
% input_mat: size channel x time x trial matrix
% win_size: window size to use
% 
% OUTPUTS
% output_mat: size channel x n_dim x trial matrix

function [output_mat] = erp_feature(input_mat, win_size)

[n_ch n_T n_trials] = size(input_mat);

n_dim = floor(n_T/win_size);

output_mat = zeros(n_ch, n_dim, n_trials);

for my_trial = 1:n_trials
    my_mat = squeeze(input_mat(:,:,my_trial));
    
    for my_dim = 1:n_dim
        my_time = [(my_dim-1)*win_size+1:win_size*my_dim];
        
        output_mat(:,my_dim,my_trial) = mean(my_mat(:,my_time),2);
    end
end