% initialize things
clear;

% static variables
% WARNING : change it manually
FILE_NAME = '../data/bwc_data.xlsx';
KNN_STR = 'KNN';
BAYES_STR = 'BAYES';
SVM_STR = 'SVM';
DTree_STR = 'DTree';
ALL_STR = 'ALL';
K_FOLD = 10;

% open xlsx and store data in an main memory array 
% each attrs values are normalized to [0,1]
data_set = data_transformation(FILE_NAME);

% start running the algoritgms
% pretty prints
disp('Choose an algorithm of the above to apply K-fold (K = 10) :');
disp('-------------------------------------');
disp('|K-Nearest Neighbor type : KNN      |');
disp('|Bayes type : BAYES                 |');
disp('|Support Vector Machines type : SVM |');
disp('|Decision Tree type : DTree         |');
disp('|For all type : ALL                 |');
disp('-------------------------------------');
fprintf('\n\n');
choice = input('Type your choice : ', 's');

% Start of algorithms
addi_info(1:5) = 0;
switch choice
    case KNN_STR
        k = input('Choose K for KNN : ');
        if k > 1
            addi_info(1) = k;
            tic;
            mdl_res = K_fold(KNN_STR, data_set, addi_info, K_FOLD);
            tm = toc;
            show_final_stats(mdl_res, tm);
        else
            disp('K for KNN <= 1, ERROR');
        end
    case BAYES_STR
        tic;
        mdl_res = K_fold(BAYES_STR, data_set, addi_info, K_FOLD);
        tm = toc;
        show_final_stats(mdl_res, tm);
    case SVM_STR
        tic;
        mdl_res = K_fold(SVM_STR, data_set, addi_info, K_FOLD);
        tm = toc;
        show_final_stats(mdl_res, tm);
    case DTree_STR
        tic;
        mdl_res = K_fold(DTree_STR, data_set, addi_info, K_FOLD);
        tm = toc;
        show_final_stats(mdl_res, tm);
    case ALL_STR
        K_fold_all(data_set, addi_info, K_FOLD);
    otherwise
        disp('ERROR on algorithm selection, please try again with valid input');
end
    