run('./startup.m')

%dataset_name = {'IP', 'PaviaU', 'PaviaC', or 'Salinas'}
dataset_name = 'IP';
[image, groundTruth, oldLabelNames] = load_dataset(dataset_name);
[im_rows, im_cols, im_bands] = size( image );
spectra = reshape( image, [im_rows*im_cols,im_bands] );
spectra = zscore(spectra,[],1);
image = reshape( spectra, [im_rows, im_cols, im_bands] );


numOfMinDataPts = 200;
numOfTrials = 20;
numTrainEg = 100;
numValEg = 50;
numTestEg = 50;
numOfLabels = max( groundTruth(:) );


labelNames = {};
labelCnt = 1;
for i = 1:numOfLabels
    locations = find(groundTruth==i);
    numOfDataPts = length( locations );
    if numOfDataPts < numOfMinDataPts
        groundTruth( locations ) = 0;
    else
        groundTruth( locations ) = labelCnt;
        labelNames{labelCnt} = oldLabelNames{i};
        labelCnt = labelCnt + 1;
    end
end
numOfLabels = labelCnt - 1;


%generate featuers
classifiers = {SVM_SE()};
classifiers_name = {'SVM_SE'};
disp( 'Computing features!!!' );
features_name = {'RAW','EMP'};
postprocess_name = {'MRF800'};

features = load (['results/MRF800_features','.mat'] );
features = features.features;

disp( 'Features computed!!!' );

numTrainEgL = [50];
for t = 1:length(numTrainEgL)
    numTrainEg = numTrainEgL(t);
    result = cell(1,numOfTrials);
    %for i =  1:numOfTrials
    for temp = 1:numOfTrials
        i = 8
       
      
        XtrainC = load(['results/MRF800_XtrainC_numTrainEgL_',num2str(t),'_numOfTrials_',num2str(i),'.mat']);
        Ytrain = load(['results/MRF800_Ytrain_numTrainEgL_',num2str(t),'_numOfTrials_',num2str(i),'.mat'] );
        XvalidC = load(['results/MRF800_XvalidC_numTrainEgL_',num2str(t),'_numOfTrials_',num2str(i),'.mat']);
        Yvalid = load(['results/MRF800_Yvalid_numTrainEgL_',num2str(t),'_numOfTrials_',num2str(i),'.mat']);       
        XtestC = load(['results/MRF800_XtestC_numTrainEgL_',num2str(t),'_numOfTrials_',num2str(i),'.mat']);
        Ytest = load(['results/MRF800_Ytest_numTrainEgL_',num2str(t),'_numOfTrials_',num2str(i),'.mat']);      
        
        XtrainC = XtrainC.XtrainC;
        Ytrain  = Ytrain.Ytrain;
        XtestC  = XtestC.XtestC;
        Ytest   = Ytest.Ytest;
        XvalidC = XvalidC.XvalidC;
        Yvalid  = Yvalid.Yvalid;         
        
        %features
        for j = 1:length(features)
            %classifiers
            for k = 1:length(classifiers)
                disp(['out_maps_numTrainEgL_',num2str(temp),'_numOfTrials_',num2str(i),'_lenFeatures_',num2str(j),'_lenClassifiers_',num2str(k),' start!']);

                [out_maps,time_taken,all_samples] = classify2(classifiers{k},image, features{j},...
                    XtrainC,Ytrain,XvalidC,Yvalid);
                disp(['out_maps_numTrainEgL_',num2str(temp),'_numOfTrials_',num2str(i),'_lenFeatures_',num2str(j),'_lenClassifiers_',num2str(k),' done!']);
                save(['results/SameProb_MRF800_double_numTrainEgL_',num2str(temp),'_numOfTrials_',num2str(i),'_lenFeatures_',num2str(j),'_lenClassifiers_', num2str(k),'.mat'], 'out_maps');
                % save MRF training performance results
                save(['logs/SameProb_MRF800_double_numTrainEgL_',num2str(temp),'_numOfTrials_',num2str(i),'_lenFeatures_',num2str(j),'_lenClassifiers_', num2str(k),'.mat'], 'all_samples');
                
                out_map = out_maps{2};
                Ypred = out_map( sub2ind([im_rows,im_cols],XtestC(:,1),XtestC(:,2)) );
                OA = nnz(Ypred==Ytest)/numel(Ypred);
                confMat = confusionmat(Ytest,Ypred);
                result{i}.(features_name{j}).(classifiers_name{k}).(postprocess_name{1}).time=time_taken(3);
                result{i}.(features_name{j}).(classifiers_name{k}).(postprocess_name{1}).OA=OA;
                result{i}.(features_name{j}).(classifiers_name{k}).(postprocess_name{1}).confMat=confMat;
                disp( [ "OA ",OA,'- time ',time_taken(3) ]);                
                disp( [ features_name{j},'-', classifiers_name{k},' done!'] );
            end
        end
        
    end
end
