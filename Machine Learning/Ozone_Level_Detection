% Compare SVM and LS-SVM (and decide on the kernel function) on the ozone 
% level detection data (see https://archive.ics.uci.edu/ml/datasets/Ozone+Level+Detection). 
% This data set contains 72 measurement variables and was measured between 1/1/1998 and 12/31/2004.
% All missing values have been removed. The goal is to detect detect whether there was too much
% ozone (class label 0) or a normal day (class label 1). The class label is the last variable. Set up
% the simulation and clearly describe what you are doing and why. Finally, state, according to your
% findings (boxplots, ROC curves, etc.), the best classifier for this problem.

% Notes:
% See ozone.xlsx for the cleaned excel version of the ozone dataset
% Go to (https://www.esat.kuleuven.be/stadius/statlssvm/?toolbox.html) for the StatLSSVM toolbox (used for lssvm model)

%%%%%%%%%%%%%%%%%% Start with Monte Carlo simulation based on 100 runs to find the mean accuracy of SVM and LS-SVM %%%%%%%%%%%%%%%%%%

data = xlsread('ozon.xlsx');
X = data(:,1:72);
Y = data(:,73);

length = size(X,1);
index0 = find(strcmp(Y,'ozone')==1);
index1 = find(strcmp(Y,'normal')==1);

% we test the mean accuracy of SVM and LS SVM based on 100 rounds 
% to find the best classifier:

 N_round = 100;
 acc_svm = zeros(N_round,1);
 acc_lssvm = zeros(N_round,1);

 for index = 1:N_round
       random_order = randperm(length);
       X = X(random_order,:);
       Y = Y(random_order);

% we take the following training/test split: 
% 1385 (3/4) training and 461 (1/4) test.       
       training_ind = 1:1385;
       testing_ind = 1385:1846;
     
 % here we use RBF for the kernel and model SVM:
       svm = fitcsvm(X(training_ind,:),Y(training_ind),'KernelFunction','RBF','OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('Optimizer', 'bayesopt','Kfold',10))
      % here we find the confusion matrix of SVM
       R = confusionmat(Y(testing_ind), predict(svm, X(testing_ind,:)));
      % here we find the accuracy of the model and find the average accuracy 
       acc_svm(index) = (R(1,1)+R(2,2))/sum(sum(R));
       
    % here we use RBF for the kernel and model LSSVM:
    model = initlssvm(X,Y,'c',[],[],'RBF_kernel');
    model = tunelssvm(model,'simplex','crossvalidatelssvm',{10,'misclass'}); 
    model = trainlssvm(model);

      % here we find the confusion matrix of LS SVM
      Rls = confusionmat(Y(testing_ind), simlssvm(model, X(testing_ind,:)));
      % here we find the accuracy of the model and find the average accuracy 
      acc_lssvm(index) = (Rls(1,1)+Rls(2,2))/sum(sum(Rls));
 end
 
 legend = {'SVM', 'LSSVM' };
   boxplot([acc_svm acc_lssvm],'Labels',legend)
   ylabel('Accuracy on test data')
   S = sprintf('SVM:%0.2f%%; LSSVM :%0.2f%%;', mean(acc_svm)*100, mean(acc_lssvm)*100 );
   disp(S);
   
 %%%%%%%%%%%%%%%%%% Determine SVM and LS-SVM ROC curve of the data %%%%%%%%%%%%%%%%%%

data = readtable('ozon.xlsx');
X = table2array(data(:,1:72));
Y = table2array(data(:,73));

% we take the following training/test split: 
% 1385 (3/4) training and 461 (1/4) test.       
       training_ind = 1:1385;
       testing_ind = 1385:1846;

% SVM ROC
% Tune an SVM classifier with RBF kernel.
svm = fitcsvm(X(training_ind,:),Y(training_ind),'KernelFunction','RBF','OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('Optimizer', 'bayesopt','Kfold',10));
mdlSVM = fitPosterior(svm);
[~,score_svm] = resubPredict(mdlSVM,X(testing_ind,:) );
%Compute the standard ROC curve using the scores from the SVM model.
[Xsvm,Ysvm,Tsvm,AUCsvm] = perfcurve(Y,score_svm(:,2),1);

%ROC curve
plot(Xsvm,Ysvm)
xlabel('False positive rate'); ylabel('True positive rate');
title('ROC Curve for SVM Classification')

% LS-SVM ROC
type = 'classification';
L_fold = 10;
[gam,sig2] = tunelssvm({X,Y,type,[],[],'RBF_kernel'},'simplex',... 
'crossvalidatelssvm',{L_fold,'misclass'}); 
[alpha,b] = trainlssvm({X,Y,type,gam,sig2,'RBF_kernel'});

% latent variables are needed to make the ROC curve
Y_latent = latentlssvm({X,Y,type,gam,sig2,'RBF_kernel'},{alpha,b},X);
[area,se,thresholds,oneMinusSpec,Sens]=roc(Y_latent,Y);
[thresholds oneMinusSpec Sens] 


%%%%%%%%%%%%%%%%%% My Conclusions %%%%%%%%%%%%%%%%%%
% Via a Monte Carlo simulation based on a hundred runs, we get that SVM has a mean accuracy of 92.94% and 
% LS-SVM has a mean accuracy of 97.02%. Through comparing the ROC curves, we can see that the overall performance 
% of LS-SVM is better than SVM. Also, notice that the boxplot of the Monte Carlo simulation shows that on average, 
% LS-SVM is better than SVM. Based on my findings, the best classifier for this problem is LS-SVM.
   
