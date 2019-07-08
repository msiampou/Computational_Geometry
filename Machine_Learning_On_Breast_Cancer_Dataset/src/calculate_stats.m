% mdl_res(1) = TP => correctly predicts possitive class
% mdl_res(2) = FP => incorrectly predicts positive class
% mdl_res(3) = FN => incorrectly predicts the negative class
% mdl_res(4) = TN => correctly predicts negative class 
function mdl_stats = calculate_stats(mdl_res, real_res)
mdl_stats(1:4) = 0; % initialize
TP_INDEX = 1;
FP_INDEX = 2;
FN_INDEX = 3;
TN_INDEX = 4;

for j = 1:length(mdl_res)
    if mdl_res(j,1) == real_res(j)
        % TP++
        if  mdl_res(j,1) == 4
            mdl_stats(TP_INDEX) = mdl_stats(TP_INDEX) + 1;
        % TN++
        else
            mdl_stats(TN_INDEX) = mdl_stats(TN_INDEX) + 1;
        end
    % mdl_res(j) ~= real_res(j)
    else
        % FP++
        if mdl_res(j,1) == 4
            mdl_stats(FP_INDEX) = mdl_stats(FP_INDEX) + 1;
        % FN++
        else
            mdl_stats(FN_INDEX) = mdl_stats(FN_INDEX) + 1;
        end
    end
end

end