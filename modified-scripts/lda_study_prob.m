function [ acc final_y final_prob X_LDA W train_set conf_matrix bias] = lda_study_prob(train_x, test_x, train_y, test_y, cov_type, bal_type)

% n_ch = length(train_x(:,1,1));
% 
% [train_mat] = erp_feature(train_x,win_size);
% [test_mat] = erp_feature(test_x,win_size);
% 
% [n_ch n_bin n_trials_train] = size(train_mat);
% [n_ch n_bin n_trials_test] = size(test_mat);

% all_predicted_y = zeros(length(test_y),n_ch);

% in all the calls, cov_type = 1, bal_type=3


train_set = train_x;
test_set = test_x;
    
[W, B, class_means] = check_lda_train_reg_auto([train_set], [train_y], cov_type, bal_type, 0);

[ X_LDA, predicted_y, acc, conf_matrix, pred_prob ] = lda_apply_prob([test_set], W, B, class_means, [test_y]);    

bias = B;

% can use the predicted_y returned by lda_apply

final_prob = pred_prob;

% disp(predicted_y)
% disp(test_y)

final_y = predicted_y;
