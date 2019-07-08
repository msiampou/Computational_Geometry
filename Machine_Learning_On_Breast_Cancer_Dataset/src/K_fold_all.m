% data_set is the whole dataset
% addi_info contains some infos for some classifiers(if we need it)
% k = the number of folds
% runs all the classifiers back to back 
function K_fold_all(data_set, addi_info, K_FOLD)
KNN_STR = 'KNN';
BAYES_STR = 'BAYES';
SVM_STR = 'SVM';
DTree_STR = 'DTree';

[rows, cols] = size(data_set);
X = data_set(:, 1:cols-1); %contains attributes vectors
Y = data_set(:, cols); %contains labels {2-> Benign, 4-> Malignant}

indices = crossvalind('Kfold', Y, K_FOLD); %geenrate indices for the k-fold
%indices = sort(indices);

%----------------------KNN-----------------------------
mdl_str = KNN_STR;
mdl_stats_final(1:4) = 0; %intialize
addi_info(1) = 7; %k = 7 best performance
tic;
for i = 1:K_FOLD
    test = (indices == i); %if indice(i) mathces the ith k-fold loop added to test data
    train = ~test; %the rest data will be for training
    mdl = gen_model(mdl_str, X(train, :), Y(train, :), addi_info); % gen = fits model with the traning partitioned dataset
    mdl_res = predict(mdl, X(test, :)); % classify, on test data
    mdl_stats_curr = calculate_stats(mdl_res, Y(test, :)); %get stats for the metrics
    mdl_stats_final = mdl_stats_final + mdl_stats_curr; % update the results
end
tm = toc;
fprintf('---KNN STATS---');
show_final_stats(mdl_stats_final, tm);
fprintf('\n');

%----------------------BAYES-----------------------------
mdl_str = BAYES_STR;
mdl_stats_final(1:4) = 0; %intialize
tic;
for i = 1:K_FOLD
    test = (indices == i); %if indice(i) mathces the ith k-fold loop added to test data
    train = ~test; %the rest data will be for training
    mdl = gen_model(mdl_str, X(train, :), Y(train, :), addi_info); % gen = fits model with the traning partitioned dataset
    mdl_res = predict(mdl, X(test, :)); % classify, on test data
    mdl_stats_curr = calculate_stats(mdl_res, Y(test, :)); %get stats for the metrics
    mdl_stats_final = mdl_stats_final + mdl_stats_curr; % update the results
end
tm = toc;
fprintf('---BAYES STATS---\n');
show_final_stats(mdl_stats_final, tm);
fprintf('\n');

%----------------------SVM-----------------------------
mdl_str = SVM_STR;
mdl_stats_final(1:4) = 0; %intialize
tic;
for i = 1:K_FOLD
    test = (indices == i); %if indice(i) mathces the ith k-fold loop added to test data
    train = ~test; %the rest data will be for training
    mdl = gen_model(mdl_str, X(train, :), Y(train, :), addi_info); % gen = fits model with the traning partitioned dataset
    mdl_res = predict(mdl, X(test, :)); % classify, on test data
    mdl_stats_curr = calculate_stats(mdl_res, Y(test, :)); %get stats for the metrics
    mdl_stats_final = mdl_stats_final + mdl_stats_curr; % update the results
end
tm = toc;
fprintf('---SVM STATS---');
show_final_stats(mdl_stats_final, tm);
fprintf('\n');

%----------------------DTree-----------------------------
mdl_str = DTree_STR;
mdl_stats_final(1:4) = 0; %intialize
tic;
for i = 1:K_FOLD
    test = (indices == i); %if indice(i) mathces the ith k-fold loop added to test data
    train = ~test; %the rest data will be for training
    mdl = gen_model(mdl_str, X(train, :), Y(train, :), addi_info); % gen = fits model with the traning partitioned dataset
    mdl_res = predict(mdl, X(test, :)); % classify, on test data
    mdl_stats_curr = calculate_stats(mdl_res, Y(test, :)); %get stats for the metrics
    mdl_stats_final = mdl_stats_final + mdl_stats_curr; % update the results
end
tm = toc;
fprintf('---DTree STATS---');
show_final_stats(mdl_stats_final, tm);
fprintf('\n');

end
