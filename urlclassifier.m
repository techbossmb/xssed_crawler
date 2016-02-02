%URLClassifier classifies url as vunerable or safe
%Positive labels (vunerable urls) were crawler from xssed.com. xssed
%    maintains a list of vunerable urls on their archive page
%Negative labels (good urls) were crawler using breadth first search by
%    following links on legitimate sites
% crawler can be found on my github account@ https://github.com/techbossmb/xssed_crawler
% 
%@author: Ishola Babatunde

%read data file
[chars, scriptCount, encodedChars, jsEvents, domEvents, label] = textread('dataset.txt', '%d%d%d%d%d%d', 'delimiter',',:/');
% randomize data
rndIndex = randperm(length(chars));
chars = chars(rndIndex);
scriptCount = scriptCount(rndIndex);
encodedChars = encodedChars(rndIndex);
jsEvents = jsEvents(rndIndex);
domEvents = domEvents(rndIndex);
label = label(rndIndex);
% build features
features = [chars, scriptCount, encodedChars, jsEvents, domEvents];
% specify training and test data
percentTraining = 0.8;
trLastIndex = floor(percentTraining*size(features, 1));
testStartIndex = trLastIndex+1;

featuresTrain = features(1:trLastIndex, :);
labelTrain = label(1:trLastIndex);
featuresTest = features(testStartIndex:end, :);
labelTest = label(testStartIndex:end);

%build model tree
tree = fitctree(featuresTrain, labelTrain);

%evaluate performance
[~, scores] = resubPredict(tree);
[X, Y, T, ~, optROC] = perfcurve(labelTrain, scores(:,2:end), 1);
figure;
plot(X,Y), hold on;
plot(optROC(1), optROC(2),'ro');
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curve for URL Vulnerability Classification');
%saveas(gcf, 'ROC curve');%
predicted = predict(tree, featuresTest);
performance = classperf(labelTest, predicted)
publish('URLClassifier.m');
