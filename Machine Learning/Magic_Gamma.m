% Consider the “Magic Gamma Telescope” data set (http://archive.ics.uci.edu/ml/datasets/MAGIC+Gamma+Telescope).
% The data are MC generated to simulate registration of high energy gamma particles in a 
% ground- based atmospheric Cherenkov gamma telescope using the imaging technique. Cherenkov 
% gamma telescope observes high energy gamma rays, taking advantage of the radiation emitted 
% by charged particles produced inside the electromagnetic showers initiated by the gammas, and 
% developing in the atmosphere. This Cherenkov radiation (of visible to UV wavelengths) leaks 
% through the atmosphere and gets recorded in the detector, allowing reconstruction of the shower
% parameters. 

%The data set consists out of 10 variables and 2 classes (last column of the data set);
% class -1 and hadron (background) in- dicated as class 1.

% Goal: Find a suitable classifier that predicts gamma particles (signal).
% Can we beat the 86.6% mean accuracy on the test data?

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Load the data and separate features from classification
gamma = readtable('magic04.data.txt');
X = table2array(gamma(:, 1:10));
y = table2array(gamma(:,11));

%%
% Check if the data is multivariate normal per class
% This will determine if LDA/QDA assumptions hold

% Gamma Class
gammaList = find(strcmp( y, 'g') == 1);
HZmvntest( X(gammaList,:) );

% Hadron Class
hadronList = find(strcmp( y, 'h') == 1);
HZmvntest( X(hadronList,:) );

% Notice that the p value for both gamma and hadron are 0,
% meaning that the data does not have a multi-variable normal distribution.
% Therefore LDA and QDA may not perform well


%% Train the various models
% Models: LDA, QDA, NB, SVM, and KNN
% 100 runs with 13000 train and 6020 test
runs = 100;

% k value for KNN
K = 1:25;

% Initialize vector/matrix for accuracy 
ldaAcc = zeros(runs, 1);
qdaAcc = zeros(runs, 1);
knnAcc = zeros(runs, size(K,2));
nbAcc = zeros(runs, 1);
svmAcc = zeros(runs, 1);

% Monte Carlo Simulation
for round = 1:runs
    randRows = randperm(length(X));
    X = X(randRows, :);
    y = y(randRows, 1);
    
    train = 1:13000;
    test = 13001:length(X);
    
    lda = fitcdiscr(X(train , :), y(train), 'DiscrimType', 'linear');
    R = confusionmat(y(test) , predict(lda, X(test,:)));
    ldaAcc(round) =  (R(1,1) + R(2,2)) /sum(sum(R));
    
    qda = fitcdiscr(X(train , :), y(train), 'DiscrimType', 'quadratic');
    R = confusionmat(y(test) , predict(qda, X(test,:)));
    qdaAcc(round) =  (R(1,1) + R(2,2)) /sum(sum(R));
    
    nb = fitcnb(X(train , :), y(train), 'Distribution', 'normal');
    R = confusionmat(y(test) , predict(nb, X(test,:)));
    nbAcc(round) =  (R(1,1) + R(2,2)) /sum(sum(R));
    
    svm = fitcsvm(X(train , :), y(train), 'KernelFunction', 'rbf', 'Standardize', true, 'ClassNames',{'g', 'h'});
    R = confusionmat(y(test) , predict(svm, X(test,:)));
    svmAcc(round) =  (R(1,1) + R(2,2)) /sum(sum(R));
    
    for k = K
        knn = fitcknn( X(train , :), y(train), 'NumNeighbors' , k, 'Standardize' , 1);
        R = confusionmat( y(test), predict( knn, X(test,:)));
        knnAcc(round , k) = (R(1,1) + R(2,2)) /sum(sum(R));
    end
end
       
[value , index ] = max(mean(knnAcc));
legend = {'LDA', 'QDA' , 'NB' , 'SVM' , 'KNN' };
boxplot( [ldaAcc qdaAcc nbAcc svmAcc knnAcc(:, index)], 'Labels', legend)
xlabel('Model')
ylabel('Test Data Accuracy')

S = sprintf( 'LDA:%0.2f%%; QDA:%0.2f%%; Naive Bayes: %0.2f%%; SVM:%0.2f%%; KNN:%0.2f%%;' ...
, mean(ldaAcc)*100, mean(qdaAcc)*100, mean(nbAcc)*100, mean(svmAcc)*100, mean(knnAcc(:, index))*100);

disp(S);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% The results give the following accuracy: LDA:78.50%; QDA:78.48%; Naive Bayes: 72.78%; SVM:86.51%; KNN:83.81%;
% Therefore SVM has the best accuracy! 


