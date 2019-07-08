% this function takes an mdl_str of the following : 
% KNN_STR = "KNN";
% BAYES_STR = "BAYES";
% SVM_STR = "SVM";
% DTree_STR = "DTree";
% (will use it for the generation of the model)

% data_set is the whole dataset
% addi_info contains some infos for some classifiers(if we need it)
% k = the number of folds
function mdl_stats_final = K_fold(mdl_str, data_set, addi_info, K_FOLD)

[rows, cols] = size(data_set);
X = data_set(:, 1:cols-1); %contains attributes vectors
Y = data_set(:, cols); %contains labels {2-> Benign, 4-> Malignant}

indices = crossvalind('Kfold', Y, K_FOLD); %geenrate indices for the k-fold
%indices = sort(indices);
mdl_stats_final(1:4) = 0; %intialize

for i = 1:K_FOLD
    test = (indices == i); %if indice(i) mathces the ith k-fold loop added to test data
    train = ~test; %the rest data will be for training
    mdl = gen_model(mdl_str, X(train, :), Y(train, :), addi_info); % gen = fits model with the traning partitioned dataset
    mdl_res = predict(mdl, X(test, :)); % classify, on test data
    mdl_stats_curr = calculate_stats(mdl_res, Y(test, :)); %get stats for the metrics
    mdl_stats_final = mdl_stats_final + mdl_stats_curr; % update the results
end



end
