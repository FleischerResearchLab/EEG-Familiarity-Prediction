% perform LDA on some data
% based on "Fisher Linear Discriminant Analysis" -- Max Welling
% and misc. random stuff I found on the internet ...
%
% INPUT
% x - design matrix -- row vectors
% y - data set outputs, assumed to be passed in as a single column vector
%     and consist of classes 0 and 1
%
% OUTPUT
% w  - weight vector
%
% 09/28/07 -- created
% 02/17/09 -- modified for current classification format
%
% function w = train_LDA(X, Y)
function [W B class_mv] = check_lda_train_reg_auto(X, Y, cov_type, bal_type, k_reg)
%% cov_type, bal_type = 1, 3

% get the list of classes
class_list = unique(Y);

% extract a few useful things
% get the index of the negative and positive class
ind0 = find(Y == class_list(2));
ind1 = find(Y == class_list(1));
num0 = length(ind0);
num1 = length(ind1);

% first find the mean for each class
m0 = mean(X(ind0, :), 1)';
m1 = mean(X(ind1, :), 1)';

% compute the within-class scatter matrices
% be lazy -- use cov and multiply by class count
% NOTE: need to nomalize by n, not n - 1 for this to work ...
if cov_type==1 % weighted cov, used in all calls.

  % zero-centered the dataset
  x0 = X(ind0,:) - repmat(m0',num0,1);
  x1 = X(ind1,:) - repmat(m1',num1,1);
  
  % concat
  new_X = cat(1,x0,x1);
  % calculate the cov matrix
  S = cov(new_X,1);
  % Changed

  k_reg = cal_shrinkage(new_X,[],1);
  % Changed by Scott
  disp("k_reg Change to 0, to turn-off the shrinkage")
  k_reg = 0;
  k_d = mean(diag(S));
  S_W = (1-k_reg)*S + eye(size(S,1))*k_reg*k_d;


elseif cov_type==2 % unweighted cov
  S_0 = cov(X(ind0, :), 1);
  S_1 = cov(X(ind1, :), 1);
  k_reg0 = cal_shrinkage(X(ind0,:),[],1);
  k_reg1 = cal_shrinkage(X(ind1,:),[],1);
  k_d0 = mean(diag(S_0));
  k_d1 = mean(diag(S_1));
  
  [dim1, dim2] = size(S_0);
  S_W = (1-k_reg0)*S_0 + eye(dim1) * k_reg0 * k_d0 + (1-k_reg1)*S_1 + eye(dim1) * k_reg1 * k_d1;
elseif cov_type==3 % unweighted cov, larger data set for shrinkage parameter
  x0 = X(ind0,:) - repmat(m0',num0,1);
  x1 = X(ind1,:) - repmat(m1',num1,1);
  
  new_X = cat(1,x0,x1);
  k_reg = cal_shrinkage(new_X,[],1);
  S_0 = cov(X(ind0, :), 1);
  S_1 = cov(X(ind1, :), 1);
  [dim1, dim2] = size(S_0);
  k_d0 = mean(diag(S_0));
  k_d1 = mean(diag(S_1));
  S_W = (1-k_reg)*S_0 + eye(dim1) * k_reg * k_d0 + (1-k_reg)*S_1 + eye(dim1) * k_reg * k_d1;
elseif cov_type==4 % weighted cov, class shrinkage
  S_0 = cov(X(ind0, :), 1);
  S_1 = cov(X(ind1, :), 1);
  k_reg0 = cal_shrinkage(X(ind0,:),[],1);
  k_reg1 = cal_shrinkage(X(ind1,:),[],1);
  k_d0 = mean(diag(S_0));
  k_d1 = mean(diag(S_1));
  
  [dim1, dim2] = size(S_0);
  S_W = num0 * ((1-k_reg0)*S_0 + eye(dim1) * k_reg0 * k_d0) + ...
    num1 * ((1-k_reg1)*S_1 + eye(dim1) * k_reg1 * k_d1);
  S_W = S_W/(num0 + num1);
end


% solve for optimal projection
W = pinv(S_W) * (m0 - m1);

B = (m0'*W+m1'*W)/2;

% change by Scott,
disp("!!!Match the Bias calculation by Sklearn!!!");
B = -(m0'*W+m1'*W)/2 + log(num0/num1);

% disp("!!!! Change Bias to Large Numbers B=1000")
% B = 1000
% disp(B)

x0 = X(ind0,:);
x1 = X(ind1,:);
if bal_type == 1 % over-sampling
  if num0 >= num1
    multiplier = floor(num0/num1);
    x1 = repmat(x1, multiplier,1);
    x1 = cat(1, x1, x1(randperm(num1, num0-num1*multiplier),:));
    X_bal = cat(1, x0, x1);
    ind1 = num0 + ind0;
  else
    multiplier = floor(num1/num0);
    x0 = repmat(x0, multiplier,1);
    x0 = cat(1, x0, x0(randperm(num0, num1-num0*multiplier),:));
    X_bal = cat(1, x0, x1);
    ind0 = 1:num1;
    ind1 = num1 + ind0;   
  end
elseif bal_type == 2 % under-sampling
  if num0 >= num1
    x0 = x0(randperm(num0,num1),:);
  else
    x1 = x1(randperm(num1,num0),:);
  end
  X_bal = cat(1, x0, x1);
  ind0 = 1:min(num0,num1);
  ind1 = min(num0,num1) + ind0;
elseif bal_type ==3  % unbalanced


  X_bal = X;
  
  ind0 = find(Y == class_list(2));
  ind1 = find(Y == class_list(1));

  
end

% project the data onto this line
% X_LDA = w' * X';
X_LDA = X_bal  * W;  % same as (w' * X')'
% size(X_bal)
% size(X)


X_LDA = X_LDA-B;

% only changes mean but not std
class_mv(1,1) = mean(X_LDA(ind0,:));
class_mv(1,2) = mean(X_LDA(ind1,:));
class_mv(2,1) = std(X_LDA(ind0,:));
class_mv(2,2) =  std(X_LDA(ind1,:));

disp(size(class_mv(2,1)));
disp(size(class_mv(2,2)));
