clc; clear all;

% Create imagesc plot of gridded 2D data
% imagesc is useful because it only plots the data points you have
% unlike contourf, which interpolates between the points.
% thus imagesc is best for plotting matrix contents in color

% importdata(filename, delimiter, num_headerlines)

%presicion
% results1 = load('results/exp3_result_IP_50.mat');
% results2 = load('results/result_IP_50.mat'); 
% time = [];
% for t = 0:1:29
%   T = round(t*1) + 1;
%   speed1(T)=results1.result{1,T}.EMP.SVM_SE.MRF800.OA;
%   speed2(T)=results2.result{1,T}.EMP.SVM_SE.MRF800.OA;
%   speed3(T)=results1.result{1,T}.EMP.SVM_SE.MRF800.time;
%   speed4(T)=results2.result{1,T}.EMP.SVM_SE.MRF800.time;
% end
% 
% 
% f=figure(1);
% hold on;
% plot(0:1:29, speed1, '-', 'color', [0 0 0], 'linewidth', 2);
% plot(0:1:29, speed2, '--', 'color', 'r', 'linewidth', 2);
% 
% xlabel('X');
% ylabel('Y');
% title('Line Plot of Column Data');
% grid on; 
% legend({'Gibbs Sampling', 'Graph Cut'}, 'location', 'southeast');
% grid off;
% hold off;

% f=figure(2);
% hold on;
% plot(0:1:29, speed3, '-', 'color', [0 0 0], 'linewidth', 2);
% plot(0:1:29, speed4, '--', 'color', 'r', 'linewidth', 2);
% 
% xlabel('X');
% ylabel('Y');
% title('time');
% grid on; 
% legend({'Gibbs Sampling', 'Graph Cut'}, 'location', 'southeast');
% grid off;
% hold off;


% M = load('matlab.mat');
% 
% outMaps = M.groundTruth;

% figure(3);
% %imagesc(x, y, Z)
% % This will work for any dimension matrix Z (it doesn't have to be square)
% % provided that it matches the dimensions of X and Y.
% % Note that unlike contour and pcolor, imagesc does NOT require a meshgrid
% % (x and y are just vectors, but M is a matrix)
% x = 0:1:144;
% y = 0:1:144;
% imagesc(x, y, outMaps);
% 
% % by default, the imagesc plot will be flipped upside-down
% % set YDir to normal to fix this
% set(gca,'YDir','normal');
% 
% xlabel('X');
% ylabel('Y');
% title(['Imagesc Plot of Groundtruth']);
% 
% caxis([0 13]);
% colormap(hot);
% ch = colorbar;
% set(ch, 'YTick', [0 13]);
% set(get(ch, 'ylabel'), 'string', 'Z');



% for i=1:1:3
%     outMaps = M.out_maps{i};
%     
%     figure(3+i);
%     %imagesc(x, y, Z)
%     % This will work for any dimension matrix Z (it doesn't have to be square)
%     % provided that it matches the dimensions of X and Y.
%     % Note that unlike contour and pcolor, imagesc does NOT require a meshgrid
%     % (x and y are just vectors, but M is a matrix)
%     x = 0:1:144;
%     y = 0:1:144;
%     imagesc(x, y, outMaps);
%     
%     % by default, the imagesc plot will be flipped upside-down
%     % set YDir to normal to fix this
%     set(gca,'YDir','normal');
%     
%     xlabel('X');
%     ylabel('Y');
%     title(['Imagesc Plot of Gibbs sampling:', i]);
%     
%     caxis([0 13]);
%     colormap(hot);
%     ch = colorbar;
%     set(ch, 'YTick', [0 13]);
%     set(get(ch, 'ylabel'), 'string', 'Z');
% end
dataList=1:1:3;
dataType='double';
% 
% for n = dataList
%     t = 1;
%     i = n;
%     j = 1;
%     k = 1;
%     im_rows = 145;
%     im_cols = 145;
%     M = load(['results/exp3_MRF800_',dataType,'_numTrainEgL_',num2str(t),'_numOfTrials_',num2str(i),'_lenFeatures_',num2str(j),'_lenClassifiers_1.mat']);
%     Xtest = load(['results/MRF800_XtestC_numTrainEgL_1_numOfTrials_',num2str(i),'.mat']);
%     Ytest = load(['results/MRF800_Ytest_numTrainEgL_1_numOfTrials_',num2str(i),'.mat']);
%     XtestC=Xtest.XtestC;
%     Ytest=Ytest.Ytest;
%     out_map = M.out_maps{2};
%     Ypred = out_map( sub2ind([145,145],XtestC(:,1),XtestC(:,2)) );
%     OA(n) = nnz(Ypred==Ytest)/numel(Ypred);
%     confMat = confusionmat(Ytest,Ypred);
% end
% [M, I] = max(OA);
% disp([M, I]);

%training state
figure(1); hold on;
validPerf = [];
Name = [];
for n=dataList
    t = 1;
    i = n;
    j = 1;
    k = 1;
    im_rows = 145;
    im_cols = 145;
    all_samples = load(['logs/SameProb_MRF800_',dataType,'_numTrainEgL_',num2str(i),'_numOfTrials_',num2str(8),'_lenFeatures_',num2str(j),'_lenClassifiers_', num2str(k),'.mat']);
    Xtest = load(['results/MRF800_XtestC_numTrainEgL_1_numOfTrials_',num2str(i), '.mat']);
    Ytest = load(['results/MRF800_Ytest_numTrainEgL_1_numOfTrials_',num2str(i), '.mat']);
    XtestC=Xtest.XtestC;
    Ytest=Ytest.Ytest;
    all_samples = all_samples.all_samples;
    nSamples = size(all_samples, 2);
    for i =1:nSamples
        out_map = all_samples(:,i);
        Ypred = out_map( sub2ind([im_rows,im_cols],XtestC(:,1),XtestC(:,2)) );
        trainPerf(i) = nnz(Ypred==Ytest)/numel(Ypred);
    end
    txt = ['OA = ', num2str(n)]
    validPerf = [validPerf, trainPerf(1000:1099)];
    for name_i = 1:1:100
        Name = [Name; string(compose('OA_%d', n))];
    end
    plot(400:1:(nSamples-1), trainPerf, 'DisplayName', txt);
end

title("Different training datasets")
xlabel('Iteration times') 
ylabel('Accuracy')
hold off;
legend show;


figure(2);
validPerf = reshape(validPerf, [100*length(dataList),1]);
boxplot(validPerf, Name);
