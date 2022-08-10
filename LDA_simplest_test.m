function acc = LDA_simplest_test(clf, exp)

addpath('/Users/scottyang/Desktop/EEG/MemoryFinale/LOSO/Exp1')

if exp == 1

    load 'data_imbalLDA_1.mat'
    tr_order = user_tr_order_1;
    proj_score = user_prob_1;
    source_label = user_source_1;
    resp_label = user_resp_1;
    behav_feat = user_feat_1;
    
    if clf == "SN_vs_MN"
        for user = 1:length(proj_score)
            train_SNMN_cm = cell(1,2);
            pos1_idx = source_label{user} == 2 & resp_label{user}==5;
            neg1_idx = source_label{user} == 2 & resp_label{user}==4;
            pos2_idx = source_label{user} == 4 & resp_label{user}==5;
            neg2_idx = source_label{user} == 4 & resp_label{user}==4;
            pos_idx = [behav_feat{user}(pos1_idx,:);behav_feat{user}(pos2_idx,:)];
            neg_idx = [behav_feat{user}(neg1_idx,:);behav_feat{user}(neg2_idx,:)];
            train_SNMN_cm{1} = cat(1, train_SNMN_cm{1}, pos_idx);
            train_SNMN_cm{2} = cat(1, train_SNMN_cm{2}, neg_idx);
            if user == 1
                train_set = cat(1, train_SNMN_cm{1}, train_SNMN_cm{2});
                train_y = [ones(size(train_SNMN_cm{1},1),1); -1*ones(size(train_SNMN_cm{2},1),1)];
            else 
                train_set = cat(1, train_SNMN_cm{1}, train_SNMN_cm{2}, train_set);
                sub_y = [ones(size(train_SNMN_cm{1},1),1); -1*ones(size(train_SNMN_cm{2},1),1)];
                train_y = cat(1, sub_y, train_y);
            end
        end
    else
        for user = 1:length(proj_score)
            train_SNMN_cm = cell(1,2);
            pos1_idx = source_label{user} == 1 & resp_label{user}==3;
            neg1_idx = source_label{user} == 2 & resp_label{user}==4;
            pos2_idx = source_label{user} == 3 & resp_label{user}==3;
            neg2_idx = source_label{user} == 2 & resp_label{user}==5;
            pos_idx = [behav_feat{user}(pos1_idx,:);behav_feat{user}(pos2_idx,:)];
            neg_idx = [behav_feat{user}(neg1_idx,:);behav_feat{user}(neg2_idx,:)];
            train_SNMN_cm{1} = cat(1, train_SNMN_cm{1}, pos_idx);
            train_SNMN_cm{2} = cat(1, train_SNMN_cm{2}, neg_idx);
            if user == 1
                train_set = cat(1, train_SNMN_cm{1}, train_SNMN_cm{2});
                train_y = [ones(size(train_SNMN_cm{1},1),1); -1*ones(size(train_SNMN_cm{2},1),1)];
            else 
                train_set = cat(1, train_SNMN_cm{1}, train_SNMN_cm{2}, train_set);
                sub_y = [ones(size(train_SNMN_cm{1},1),1); -1*ones(size(train_SNMN_cm{2},1),1)];
                train_y = cat(1, sub_y, train_y);
            end
        end
    end
% exp 2
else 
    load 'data_imbalLDA_2.mat'
    tr_order = user_tr_order_2;
    proj_score = user_prob_2;
    source_label = user_source_2;
    resp_label = user_resp_2;
    behav_feat = user_feat_2;
%%
    if clf == "SN_vs_MN"
        for user = 1:length(proj_score)
            train_SNMN_cm = cell(1,2);
            pos1_idx = source_label{user} == 2 & resp_label{user}==5;
            neg1_idx = source_label{user} == 2 & resp_label{user}==4;
            pos2_idx = source_label{user} == 4 & resp_label{user}==5;
            neg2_idx = source_label{user} == 4 & resp_label{user}==4;
            pos_idx = [behav_feat{user}(pos1_idx,:);behav_feat{user}(pos2_idx,:)];
            neg_idx = [behav_feat{user}(neg1_idx,:);behav_feat{user}(neg2_idx,:)];
            train_SNMN_cm{1} = cat(1, train_SNMN_cm{1}, pos_idx);
            train_SNMN_cm{2} = cat(1, train_SNMN_cm{2}, neg_idx);
            if user == 1
                train_set = cat(1, train_SNMN_cm{1}, train_SNMN_cm{2});
                train_y = [ones(size(train_SNMN_cm{1},1),1); -1*ones(size(train_SNMN_cm{2},1),1)];
            else 
                train_set = cat(1, train_SNMN_cm{1}, train_SNMN_cm{2}, train_set);
                sub_y = [ones(size(train_SNMN_cm{1},1),1); -1*ones(size(train_SNMN_cm{2},1),1)];
                train_y = cat(1, sub_y, train_y);
            end
        end
    else
        for user = 1:length(proj_score)
            train_SNMN_cm = cell(1,2);
            pos1_idx = source_label{user} == 1 & resp_label{user}==3;
            neg1_idx = source_label{user} == 2 & resp_label{user}==4;
            pos2_idx = source_label{user} == 3 & resp_label{user}==3;
            neg2_idx = source_label{user} == 2 & resp_label{user}==5;
            pos_idx = [behav_feat{user}(pos1_idx,:);behav_feat{user}(pos2_idx,:)];
            neg_idx = [behav_feat{user}(neg1_idx,:);behav_feat{user}(neg2_idx,:)];
            train_SNMN_cm{1} = cat(1, train_SNMN_cm{1}, pos_idx);
            train_SNMN_cm{2} = cat(1, train_SNMN_cm{2}, neg_idx);
            if user == 1
                train_set = cat(1, train_SNMN_cm{1}, train_SNMN_cm{2});
                train_y = [ones(size(train_SNMN_cm{1},1),1); -1*ones(size(train_SNMN_cm{2},1),1)];
            else 
                train_set = cat(1, train_SNMN_cm{1}, train_SNMN_cm{2}, train_set);
                sub_y = [ones(size(train_SNMN_cm{1},1),1); -1*ones(size(train_SNMN_cm{2},1),1)];
                train_y = cat(1, sub_y, train_y);
            end
        end
    end
end
disp("The shape of the training set is")
disp(size(train_set))

[acc, final_pred2, output_prob, output_proj, W, train_set] = lda_study_prob(train_set, train_set, train_y, train_y, 2, 3);

out_msg = strcat("the accuracy for experiment " + exp + " " + clf + " is: " + acc);

disp(out_msg)