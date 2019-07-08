% gets all the TP,TN,TF,FN

% reminder here : 
% mdl_res(1) = TP => correctly predicts possitive class
% mdl_res(2) = FP => incorrectly predicts positive class
% mdl_res(3) = FN => incorrectly predicts the negative class
% mdl_res(4) = TN => correctly predicts negative class 
% tm => total_time (how long k-fold took us)

% caclulates accuracy, sensitivity, scecificity
function show_final_stats(f_stats, tm)
TP_INDEX = 1;
FP_INDEX = 2;
FN_INDEX = 3;
TN_INDEX = 4;
TOTAL_PREDICTED = sum(f_stats);

accuracy = (f_stats(TP_INDEX) + f_stats(TN_INDEX))/TOTAL_PREDICTED;
sensitivity = f_stats(TP_INDEX)/(f_stats(TP_INDEX) + f_stats(FN_INDEX));
specificity = f_stats(TN_INDEX)/(f_stats(TN_INDEX) + f_stats(FP_INDEX));

fprintf('\n-----------------------------------------\n');
fprintf('|Total Predicted = %d, K-fold with K = 10 \n', TOTAL_PREDICTED);
fprintf('|Total time for K-fold = %f secs           \n', tm);
fprintf('|Total True Positive   = %d                \n', f_stats(TP_INDEX));
fprintf('|Total True Negative   = %d                \n', f_stats(TN_INDEX));
fprintf('|Total False Positive  = %d                \n', f_stats(FP_INDEX));
fprintf('|Total False Negative  = %d                \n', f_stats(FN_INDEX));
fprintf('|Accuracy              = %.3f%%            \n', accuracy*100);
fprintf('|Sensitivity           = %.3f%%            \n', sensitivity*100);
fprintf('|Specificity           = %.3f%%            \n', specificity*100);
fprintf('------------------------------------------\n');

end