% reads xlxs file
% note : has replaced ? with most freq vals
% will normalize data to range of [0,1]
% returns array with the whole data set
function data_set = data_transformation(FILE_NAME)
data_set = xlsread(FILE_NAME);
[rows, cols] = size(data_set);

%---------my normalize----------
% normalize dataset to range of [0,1]
% find max value for every column ( O(n))
% as a result each attr's( = column's) values have been normalized to [0,1]
for j = 1:cols-1
    %------------my normalize-----------
    curr_col = data_set(:,j);
    max_val_col = max(curr_col);
    min_val_col = min(curr_col);
    curr_col = (curr_col-min_val_col)/(max_val_col-min_val_col);
    data_set(:,j) = curr_col;
end

%---------matlabs normalize----------
%---------changed matlab to apply this method-------
%data_set(:,1:cols-1) = normalize(data_set(:,1:cols-1) ,'range');

end