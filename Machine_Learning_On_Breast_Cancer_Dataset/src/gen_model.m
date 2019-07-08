% this function takes an str of the following : 
% KNN_STR = "KNN";
% BAYES_STR = "BAYES";
% SVM_STR = "SVM";
% DTree_STR = "DTree";

% generates an classifier
% addi_info(1) = k if we choose knn contains the centers

% X = containts instances of the dataset ( vector of values )
% Y = contains the last col => classes
function mdl = gen_model(mdl_str, X, Y, addi_info)
% check which method we run
switch mdl_str
    case 'KNN'
        % info for the KNN
        % default uses euclidean distance
        % default distance is the best 
        % distance weights inverse or equal are the best
        % tie breakers smallest and nearest are the best
        mdl = fitcknn(X, Y, 'NumNeighbors', addi_info(1));
        
        % ---------Distances--------
        mdl.Distance = 'euclidean';
        % KNN with minkowski distance
        %mdl.Distance = 'minkowski';
        % KNN with cityblock distance 
        %mdl.Distance = 'cityblock';
        % KNN with chebychev distance
        %mdl.Distance = 'chebychev';
           
        % --------Weights----------
        %mdl.DistanceWeight = 'equal';
        % distance = 1/dist
        mdl.DistanceWeight = 'inverse';
        % distance = 1/dist^2
        %mdl.DistanceWeight = 'squaredinverse';
       
        % -------Ties for class costs--------
        mdl.BreakTies = 'nearest';
        %mdl.BreakTies = 'smallest';
        %mdl.BreakTies = 'random';
        
        %--------Cost of misclassification------
        % Square matrix, where Cost(i,j) is the cost of classifying a point into 
        % class j if its true class is i
        mdl.Cost = [0 1; 4 0]; % best cost found from testing
        %mdl.Cost = [0 0.2; 1 0];
        %mdl.Cost = [0 0.25; 0.8 0];
        
    case 'BAYES'
        % basic model of bayes
        %mdl = fitcnb(X, Y); 
       
        % all the smooth kernels
        %mdl = fitcnb(X, Y,'DistributionNames','kernel', 'Kernel', 'normal');
        %mdl = fitcnb(X, Y,'DistributionNames','kernel', 'Kernel', 'epanechnikov');
        %mdl = fitcnb(X, Y,'DistributionNames','kernel', 'Kernel', 'box');
        %mdl = fitcnb(X, Y,'DistributionNames','kernel', 'Kernel', 'triangle');
        
        % mvmn, normal
        mdl = fitcnb(X, Y,'CategoricalPredictors', 'all', 'DistributionNames', 'mvmn');
        %mdl = fitcnb(X, Y,'DistributionNames','normal');
        
        
        % mini optimization because we know from the problem
        % Class distribution:
        % 1) Benign: 458 (65.5%)
        % 2) Malignant: 241 (34.5%)
        % doesn't give a lot of speed up
        mdl.Prior = struct('ClassNames',[2, 4],'ClassProbs',[0.655; 0.345]);
        %mdl.Cost = [0 0.1; 0.9 0];
        %mdl.Cost = [0 0.2; 0.8 0];
        mdl.Cost = [0 1; 4 0];
    case 'SVM'
        % here we cant set cost
        % default use SMO as SOLVER
        % ------ Just using kernel functions ---------
        mdl = fitcsvm(X, Y, 'Cost', [0 1; 4 0], 'KernelFunction', 'gaussian'); %K(x,x') = exp(-||x-x'||^2)
        %mdl = fitcsvm(X, Y, 'Cost', [0 1; 4 0], 'KernelFunction', 'linear'); %K(x,x') = x^t * x'

        % ------- Polys with q------
        %mdl = fitcsvm(X, Y, 'Cost', [0 1; 4 0], 'KernelFunction', 'polynomial', 'PolynomialOrder', 2); %K(x,x') = (1+x^t*x')^q, here q = 2)
        %mdl = fitcsvm(X, Y, 'Cost', [0 1; 4 0], 'KernelFunction', 'polynomial', 'PolynomialOrder', 3); %K(x,x') = (1+x^t*x')^q, here q = 3)
        %mdl = fitcsvm(X, Y, 'Cost', [0 1; 4 0], 'KernelFunction', 'polynomial', 'PolynomialOrder', 4); %K(x,x') = (1+x^t*x')^q, here q = 4)
        %mdl = fitcsvm(X, Y, 'Cost', [0 1; 4 0], 'KernelFunction', 'polynomial', 'PolynomialOrder', 5); %K(x,x') = (1+x^t*x')^q, here q = 5)

        % ------ Kenrnel functions + Solver optimization ------
        %mdl = fitcsvm(X, Y, 'Cost', [0 1; 4 0], 'KernelFunction', 'gaussian', 'Solver', 'SMO', 'CacheSize','maximal');
        %mdl = fitcsvm(X, Y, 'Cost', [0 1; 4 0], 'KernelFunction', 'gaussian', 'Solver', 'ISDA', 'CacheSize','maximal'); %K(x,x') = exp(-||x-x'||^2)
        %mdl = fitcsvm(X, Y, 'Cost', [0 1; 4 0], 'KernelFunction', 'gaussian', 'Solver','L1QP','CacheSize','maximal'); %K(x,x') = exp(-||x-x'||^2)
        
        %---Solvers dosen't optimize kernelfunctions linear and polynomial
        % as the gaussian
        %mdl = fitcsvm(X, Y, 'Cost', [0 1; 4 0], 'KernelFunction', 'linear', 'Solver','ISDA'); %K(x,x') = exp(-||x-x'||^2)
        %mdl = fitcsvm(X, Y, 'Cost', [0 1; 4 0], 'KernelFunction', 'linear', 'Solver','L1QP'); %K(x,x') = exp(-||x-x'||^2)
        %mdl = fitcsvm(X, Y,'Cost', [0 1; 4 0], 'KernelFunction', 'polynomial', 'PolynomialOrder', 2, 'Solver','ISDA'); %K(x,x') = exp(-||x-x'||^2)
        %mdl = fitcsvm(X, Y, 'Cost', [0 1; 4 0],'KernelFunction', 'polynomial', 'PolynomialOrder', 2, 'Solver','L1QP'); %K(x,x') = exp(-||x-x'||^2)
    case 'DTree'
        %mdl = fitctree(X, Y, 'Cost',  [0 1; 3 0]');
        mdl = fitctree(X, Y, 'Cost',  [0 1; 4 0], 'AlgorithmForCategorical','Exact');
        %mdl = fitctree(X, Y, 'Cost',  [0 1; 4 0], 'AlgorithmForCategorical','PullLeft');    
        %mdl = fitctree(X, Y, 'Cost',  [0 1; 4 0], 'AlgorithmForCategorical','PCA');
        %mdl = fitctree(X, Y, 'Cost',  [0 1; 4 0], 'AlgorithmForCategorical','OVAbyClass');
        
        % with split cretirion
        %mdl = fitctree(X, Y, 'SplitCriterion', 'gdi', 'Cost',  [0 1; 4 0], 'AlgorithmForCategorical','Exact');
        %mdl = fitctree(X, Y, 'SplitCriterion', 'twoing', 'Cost',  [0 1; 4 0], 'AlgorithmForCategorical','Exact');
        %mdl = fitctree(X, Y, 'SplitCriterion', 'deviance', 'Cost',  [0 1; 4 0], 'AlgorithmForCategorical','Exact');
    otherwise
       return; % will cause an error
end