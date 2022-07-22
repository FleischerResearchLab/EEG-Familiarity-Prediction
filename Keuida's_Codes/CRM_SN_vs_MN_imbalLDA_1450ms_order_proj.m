clear all
close all

user_vector={'02','03','04','05','06','08','09','10','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29'};
addpath /mnt/cube/home/lkueida/Documents/MATLAB/Memory2018/Exp1_SOSI/
load ALLDATA_1.mat

%% Classifier training
clear OLD NEW EXTRA

eeglab;
close all

time_start = 114;
time_end = 413;
time_period = time_start:time_end;

ch_cluster_index = { [19 24 28 29 20 12 13],...
        [4 5 124 118 112 117 111],...
        [7 106 31 55 80],...
        [42 37 54 53 52 61 60],...
        [87 93 79 86 78 92 85],...
        [62 67 72 77 71 76]};

for user = [1:length(user_vector)]


    MERGEDATA = ALLDATA(user).CR;
    MERGEDATA.data = cat(3,ALLDATA(user).SC.data,  ALLDATA(user).CR.data,  ALLDATA(user).SI.data,  ALLDATA(user).IR.data,  ALLDATA(user).FA.data);
    MERGEDATA.etc=[];
    MERGEDATA.chanlocs=[];
    MERGEDATA.nbchan=128;
    MERGEDATA.trials = length(MERGEDATA.data(1,1,:));
    MERGEDATA.event=[];
    MERGEDATA.icaact=[];
    MERGEDATA.epoch=[];
    MERGEDATA.specdata=[];
    MERGEDATA.icachansind=[];
    MERGEDATA.icasphere=[];
    MERGEDATA.icawinv=[];
    MERGEDATA.specicaact=[];
    MERGEDATA.setname=[];
    MERGEDATA.icaweights=[];
    temp_labels = [ones(length(ALLDATA(user).SC.resp),1);2*ones(length(ALLDATA(user).CR.resp),1);3*ones(length(ALLDATA(user).SI.resp),1); ...
        4*ones(length(ALLDATA(user).IR.resp),1);5*ones(length(ALLDATA(user).FA.resp),1)];


    % select trails for training data.
    temp_resp = [ALLDATA(user).SC.resp;ALLDATA(user).CR.resp;ALLDATA(user).SI.resp;ALLDATA(user).IR.resp;ALLDATA(user).FA.resp];

    [ temp MERGEDATA ] = preprocess_by_trial( MERGEDATA, 0, [.2], [], 40, [], [-.2 2.0]);
    [ temp MERGEDATA ] = preprocess_by_trial( MERGEDATA, 1, [.2], [.1 50], 40, [], [-.2 2.0]);
    [ temp prep_ALL ] = preprocess_by_trial( MERGEDATA, 1, [0.2], [], 20, [], [-0.2 2.0]); 


    % CR-SN vs CR-MN
    sc_index = find(temp_resp==5);
    cr_index = find(temp_resp==4);
    si_index = find(temp_resp<=3);

    extra_index = si_index;

    if(~isempty(sc_index))
      % pos class
      % NEW neg class
      OLD(user) = pop_select(prep_ALL, 'trial', sc_index, 'sorttrial', 'off');
    else
      OLD(user).data = [];
    end
    if(~isempty(cr_index))
      NEW(user) = pop_select(prep_ALL, 'trial', cr_index, 'sorttrial', 'off');
    else
      NEW(user).data = [];
    end

    EXTRA(user) = pop_select(prep_ALL, 'trial', extra_index, 'sorttrial', 'off');
    user_labels{user} = [ones(length(sc_index),1);2*ones(length(cr_index),1);3*ones(length(si_index),1)];
    user_source{user} = [temp_labels(sc_index);temp_labels(cr_index);temp_labels(si_index)];
    user_resp{user} = [temp_resp(sc_index);temp_resp(cr_index);temp_resp(si_index)];

    user_tr_order{user} = [sc_index; cr_index; extra_index];
    my_prob{user} = zeros(length(user_labels{user}),1);
end

class_min_num = zeros(1,length(user_vector));
for user = 1:length(user_vector)
  if (isempty(NEW(user).data) || isempty(OLD(user).data))
    continue
  end
  class_min_num(user) = min(size(OLD(user).data,3), size(NEW(user).data,3));
end

c_weights = cell(1,length(NEW));
test_feat = cell(1,length(NEW));
for pick = 1:length(NEW)
        
    train_OLD = [];
    train_NEW = [];
    for i = [1:pick-1 pick+1:length(NEW)]
      if(~isempty(OLD(i).data))
        train_OLD = cat(3,train_OLD, OLD(i).data(:, time_period, :));
      end
      if(~isempty(NEW(i).data))
        train_NEW =  cat(3,train_NEW, NEW(i).data(:, time_period, :));
      end
    end


    train_data{1} = cluster_channels(train_OLD, ch_cluster_index(1:end));
    train_data{2} = cluster_channels(train_NEW, ch_cluster_index(1:end));

    if(~isempty(OLD(pick).data))
      test_data{1} = cluster_channels(OLD(pick).data(:,time_period,:),ch_cluster_index(1:end));
    else
      test_data{1} = [];
    end
    if(~isempty(NEW(pick).data))
      test_data{2} = cluster_channels(NEW(pick).data(:,time_period,:),ch_cluster_index(1:end));
    else
      test_data{2} = [];
    end
    

    extra_data{1} = cluster_channels(EXTRA(pick).data(:,time_period,:),ch_cluster_index(1:end));
    extra_data{2} = extra_data{1}(:,:,1);

    train_x = cat(3, train_data{1}, train_data{2});
    test_x = cat(3, test_data{1}, test_data{2}, extra_data{1}, extra_data{2});
    train_y = [ones(size(train_data{1},3),1); -1*ones(size(train_data{2},3),1)];
    test_y = [ones(size(test_x,3)-1,1);-1];
    
    win_size = 25;
    
    [train_set] = erp_feature(train_x,win_size);
    [n_ch, n_bin, n_trials_train] = size(train_set);
    train_set = reshape(train_set, [n_bin*n_ch n_trials_train])';

    [test_set] = erp_feature(test_x,win_size);
    [~, ~, n_trials_test] = size(test_set);
    test_set = reshape(test_set, [n_bin*n_ch n_trials_test])';
    % train the LDA first,
    [acc, final_pred_2, output_prob, output_proj, W, train_set] = lda_study_prob(train_set, test_set, train_y, test_y, 1, 3);
%         [ acc final_pred_2 output_prob prob_out_pat W train_set] =  erp_feature_prob(train_x, test_x, train_y, test_y, 25, 5);    
    my_prob{pick} = output_proj(1:end-1,1);

    c_weights{pick} = W;
    
    trainsub_set = [];
    trainsub_num = [];
    for i = [1:pick-1 pick+1:length(NEW)]
      % EXTRA, everything else than pos and neg
      temp_x = [];
      if(~isempty(OLD(i).data))
        temp_x = cat(3,temp_x, OLD(i).data(:, time_period, :));
      end
      if(~isempty(NEW(i).data))
        temp_x =  cat(3,temp_x, NEW(i).data(:, time_period, :));
      end
      temp_x =  cat(3,temp_x, EXTRA(i).data(:, time_period, :));
      temp_x = cluster_channels(temp_x,ch_cluster_index(1:end));
      temp_x = erp_feature(temp_x,win_size);
      [n_ch, n_bin, n_trials_train] = size(temp_x);
      temp_x = reshape(temp_x, [n_bin*n_ch n_trials_train])';
      trainsub_set = cat(1, trainsub_set, temp_x);
      trainsub_num = cat(1, trainsub_num, i*ones(size(temp_x,1),1));
    end
    trainsub_y = [ones(length(trainsub_set)-1,1);0];
    
    % subject
    [acc, final_pred_2, output_prob, output_proj, W, train_set] = lda_study_prob(train_set, trainsub_set, train_y, trainsub_y, 1, 3);
    % projection on the test subject
%         [ acc final_pred_2 output_prob prob_out_pat W train_set] =  erp_feature_prob(train_x, test_x, train_y, test_y, 25, 5);    
    
    trainsub_prob = cell(1,length(NEW));
    for i = [1:pick-1 pick+1:length(NEW)]
      trainsub_prob{i} = output_proj(trainsub_num==i,1);
    end
    
    train_prob{pick} = trainsub_prob;
    test_feat{pick} = test_set(1:end-1,:);
end

user_source_1 = user_source;
user_resp_1 = user_resp;
user_prob_1 = my_prob;
user_class_min_1 = class_min_num;
user_tr_order_1 = user_tr_order;
user_train_prob_1 = train_prob;
user_weights_1 = c_weights;
user_feat_1 = test_feat;

save('data_CRM_SN_vs_MN_imbalLDA_order_proj_1.mat','user_prob_1','user_source_1',...
    'user_resp_1','user_class_min_1','user_tr_order_1','user_train_prob_1','user_weights_1','user_feat_1')