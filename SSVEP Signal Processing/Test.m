%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Project Name:SSVEP based BCI in controlling NAO robot
%Author:      Erxin Wang
%Date:        21.July. 2016

%%Parameters
%Input:K              test numer
%      vis            seclect channel to present
%                     1=O1  2=O2
%      EPO.start      the beginning of epoch
%      EPO.period     the duration of epoch
%      EPO.time       based on the aqusiation configuration
%                     here stimuli + rest = 10sec ==>EPO.time 

%Output:PSD           Power Spectrul Density according to Welch
%       FEAT          7,5Hz and 12Hz in O1 and O2 with a gaussian window,
%                     4 dimensional

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
close all
%% Initialization
% load EEG.mat
K = 4; %test K
vis = 1; %channel to present
rej_val = 0; %do bad trial rejection 
visualize = 1; %visualize the fishrank sorted features
%construct the EPO varible
EPO.data = [];
EPO.start = 0.5;
EPO.period = 4.5;
EPO.time = 10;
if vis ==1
    pres = 'O1';
end
if vis ==2
    pres = 'O2';
end
%1=O1 2=O2
if K==1
    filename.a = 'S1-1A-18.07.16.22.45.44.edf';
    filename.b = 'S1-1B-18.07.16.22.52.09.edf';
    filename.c = 'S1-1C-18.07.16.22.55.11.edf';
    filename.d = 'S1-1D-18.07.16.23.01.55.edf';
end
if K==2
    filename.a = 'S1-2A-19.07.16.22.25.23.edf';
    filename.b = 'S1-2B-19.07.16.22.29.16.edf';
    filename.c = 'S1-2C-19.07.16.22.32.36.edf';
    filename.d = 'S1-2D-19.07.16.22.35.39.edf';
end
if K==3
    filename.a = 'S1-3A-20.07.16.14.31.45.edf';
    filename.b = 'S1-3B-20.07.16.14.36.59.edf';
    filename.c = 'S1-3C-20.07.16.14.41.56.edf';
    filename.d = 'S1-3D-20.07.16.14.45.56.edf';
end
 if K==4
    filename.a = 'S1-A4-20.07.16.19.58.14.edf';
    filename.b = 'S1-B4-20.07.16.20.01.49.edf';
    filename.c = 'S1-C4-20.07.16.20.10.14.edf';
    filename.d = 'S1-D4-20.07.16.20.13.10.edf'; 
   filename.ref1 = 'S1-5ALPHA1-22.07.16.19.50.12.edf';
   filename.ref2 = 'S1-5ALPHA2-22.07.16.19.55.51.edf';
 end
if K ==5
    filename.a = 'S1-5A-22.07.16.19.03.49.edf';
    filename.b = 'S1-5B-22.07.16.19.06.48.edf';
    filename.c = 'S1-5C-22.07.16.19.25.43.edf';
    filename.d = 'S1-5D-22.07.16.19.29.40.edf'; 
end    
if K==0
    filename.a = 'S3-1A-18.07.16.22.22.54.edf';
    filename.b = 'S3-1B-18.07.16.22.29.04.edf';
    filename.c = 'S3-1C-18.07.16.22.32.27.edf';
    filename.d = 'S3-1D-18.07.16.22.35.27.edf';  
 end
%get the session data
[hdr1,D.a] = edfread(filename.a);
[~,D.b]    = edfread(filename.b);
[~,D.c]    = edfread(filename.c);
[~,D.d]    = edfread(filename.d);
[~,D.ref1] = edfread(filename.ref1);
[~,D.ref2] = edfread(filename.ref2);

%combine them into a whole dataset
%extrach the 14 channel
%each session has 12 trials
% M =[D.a(3:16,641:16000),D.b(3:16,641:16000),D.c(3:16,641:16000),D.d(3:16,641:16000)]; 

M =[D.a(3:16,641:16000),D.b(3:16,641:16000),D.c(3:16,641:16000),D.d(3:16,641:16000),...
    D.ref1(3:16,641:16000),D.ref1(3:16,641:16000)]; %this include the
%     reference dataset

len = size(M,2); %get the length of samples
t = linspace(0,len/128,len); %generate the time line
% lab = repmat([1 2 2 1 1 1 2 2 1 2 2 1],[1,4]); %get the labels
% trial = 48;
lab = repmat([1 2 2 1 1 1 2 2 1 2 2 1],[1,4]); %get the labels including
% reference
lab = [lab,zeros(1,12*2)];
trial = 72;
 %because there is a mistake when collecting data
 %the label in previous experiment is wrong
if K==0||K==1||K==2||K==3 
    lab = repmat([1 2 2 1 1 1 2 2 1 1 2 1],[1,4]);
end

channel = [7,8];
%7 = O1
%8 = O2

%plot the raw data with trial segmentation
figure(1)
plot(t,M(channel(1),:),t,M(channel(2),:)+100)
legend(hdr1.label{channel(1)+2},hdr1.label{channel(2)+2})
grid on
title('Row Signal');xlabel('Time(seconds)')
hold on
for i = 0:trial-1
    line([10*i,10*i],[4000,4500],'Color','red','LineWidth',0.5)
    line([10*i+5,10*i+5],[4000,4500],'Color','green','LineStyle','--','LineWidth',0.5)
end
hold off
% trial = floor((L/(128*EPO.time))); %generate the trials 

%% Preprocessing
% rejection bad trials it is based on the obsevation
% some extrem peak should be deleted

% spatial filtering 
CAR = mean(M);
for ch = 1:14
    M(ch,:) = M(ch,:)-CAR;
end

% detrending (alternative: high-pass filter with cutoff 1 Hz)
for ch = 1:14
    M(ch,:) = M(ch,:)-mean(M(ch,:),2);
end

% band pass filter
low  = 3;
high = 50;
for ch = 7:8
    M(ch,:) = bandfilter(M(ch,:),low,high,128);
end


%plot the pre-processed data
figure(2)
plot(t,M(7,:),t,M(8,:)+100)
legend(hdr1.label{channel(1)+2},hdr1.label{channel(2)+2})
grid on
title('Artifacts Removed Signal');xlabel('Time(seconds)')
hold on
for i = 0:trial-1
    line([10*i,10*i],[-100,200],'Color','red','LineWidth',0.5)
    line([10*i+5,10*i+5],[-100,200],'Color','green','LineStyle','--','LineWidth',0.5)
end
hold off

%% Signal obsevation
%construct the EEG standard matrix
%first dimension is channel space (from 1 to 14)
%secon dimension is sample space
%third dimension is trial space
for e = 1:trial
    EPO.data(:,:,e) = M(channel,(EPO.start+EPO.time*(e-1))*128+1:(EPO.start+EPO.time*(e-1)+EPO.period)*128);
end


%resolution in Frequency domain
%here the multiplier 4 stands 1:4 resolution in frequency domain
window = 4*128;
noverlap = 510;
%practically, noverlap = 0.75 should be optimal w.r.t. speed and accuracy
nfft = 4*128;
fs = 128;
%compute the PSD 
for e = 1:size(EPO.data,3)
    for ch = 1:size(EPO.data,1)
        [PSD(ch,:,e),f] = pwelch(EPO.data(ch,1:end,e),window,noverlap,nfft,fs);
    end
end
%alernative metric in dB
%PSD = 10*log(PSD);

%visulization of the SSVEP phenomenon
title_name = strcat('Channel_',pres);
figure('Name',title_name)
subplot(3,1,1)
plot(f(1:end),squeeze(PSD(vis,1:end,lab==1)))
hold on
plot(f(1:end),mean(squeeze(PSD(vis,1:end,lab==1)),2),'k','LineWidth',3)
title('The 7.5Hz signal')
% legend('Trial1','Trial2','Trial3','Trial4','Trial5','Trial6','Average')
axis([0,40,-inf,inf])
subplot(3,1,2)
plot(f(1:end),squeeze(PSD(vis,1:end,lab==2)))
hold on
plot(f(1:end),mean(squeeze(PSD(vis,1:end,lab==2)),2),'k','LineWidth',3)
title('The 12Hz signal')
% legend('Trial1','Trial2','Trial3','Trial4','Trial5','Trial6','Average')
axis([0,40,-inf,inf])
subplot(3,1,3)
plot(f(1:end),mean(squeeze(PSD(vis,1:end,lab==1)),2),'b','LineWidth',3)
hold on
plot(f(1:end),mean(squeeze(PSD(vis,1:end,lab==2)),2),'g','LineWidth',3)
title('The averaged 7.5Hz signal and 12Hz signal');xlabel('Frequency(Hz)');
legend('7.5Hz','12Hz')
axis([0,40,-inf,inf])
fprintf('Test %s Channel %s',num2str(K),pres)

% figure(4)
%visulization of time-frequency using spectrogram
% spectrogram(epochs(1,1:end,5),window,noverlap,nfft,fs,'yaxis')
% % axis([10 20 2 3 -inf 10])
% view([90,0])


%randperm(24)
%generate a 1~24 random sequence



%Above are all the contrubutions of Erxin Wang
%% Feature extrachtion
% bandpass analyse
bw = zeros(51,6);
for num=1:6
    error_of_lda = zeros(51,2);
    ww = 0;
for w = 0:0.2:10
weight_factor = 0.1*w; % 0.8 should be the optimal
ww = ww+1;
% weight factor to extrach the feature of the signal
% combined with 2 order harmonics frequency
for trial = 1:size(PSD,3)
    for ch = 1:size(PSD,1) 

        FEAT.low(trial,ch)  =(weight_factor)*PSD(ch,f==7.5,trial)+(1-weight_factor)*PSD(ch,f==15,trial);
        FEAT.high(trial,ch) = weight_factor*PSD(ch,f==12.25,trial)+(1-weight_factor)*PSD(ch,f==37,trial);
        FEAT.ref(trial,ch) = mean(PSD(ch,40:1:44,trial));
    end
end
% concatenate 7.5Hz and 12.25Hz into one matrix
FEAT.allFeatures = cat(2,FEAT.low,FEAT.high,FEAT.ref);
% FEAT.allFeatures = 1/2*(FEAT.low+FEAT.high);
FEAT.labels = lab'; %get the label vector
idx = randperm(length(FEAT.allFeatures));
FEAT.allFeatures = FEAT.allFeatures(idx,:);
FEAT.labels = FEAT.labels(idx);

%   here !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

% fenduan ######################################
% Classification/Modelling-----------LDA
lda_e = zeros(100,2);
for lda = 1:100
FEAT.labels = lab'; %get the label vector
FEAT.allFeatures = cat(2,FEAT.low,FEAT.high,FEAT.ref);
idx = randperm(length(FEAT.allFeatures));
FEAT.allFeatures = FEAT.allFeatures(idx,:);
FEAT.labels = FEAT.labels(idx);


% ONE VS ALL 

% Build 0 vs 1&2
% FEAT.labels = Label;
Label = FEAT.labels;
FEAT.label0 = Label;
FEAT.label0(FEAT.label0==2)=-1;
FEAT.label0(FEAT.label0==1)=-1;
FEAT.label0(FEAT.label0==0)=1;
[FEAT.d_all_0 FEAT.rank_all_0] = fisherrank(FEAT.allFeatures,FEAT.label0);
% change label 2 into label -1
% -1 = 7.5Hz
% 1  = 12Hz
% [FEAT.d_all_0 FEAT.rank_all_0] = fisherrank(FEAT.allFeatures,FEAT.labels);


% change label 2 into label -1
% -1 = 7.5Hz
% 1  = 12Hz
% [FEAT.d_all_1 FEAT.rank_all_1] = fisherrank(FEAT.allFeatures,FEAT.labels);

% Build 2 vs 0&1
% FEAT.label2 = Label;
% FEAT.label2(Label==2)=1;
% FEAT.label2(Label==1)=-1;
% FEAT.label2(Label==0)=-1;




% using LDA classifier instead of SVM classifier
% train with 3 best features
% MCR is not representative as we haven't predicted unseen data, therefore
% cross-validation is needed to simulate unseen data
% model = trainShrinkLDA(FEAT.allFeatures(train,:),FEAT.labels(train),0.1,'');
% lamda =0.1
% yhat = predictShrinkLDA(model,FEAT.allFeatures(test,:));
% MCR(fold) = sum(FEAT.labels(test) ~= yhat')/length(FEAT.labels(test));

%using K-fold Cross-Validation
%Bi.Wang's code is here
% Cross validation for 0&(1,2)
    FEAT.labels = Label;
    nFolds = 8;
    indices = crossvalind('Kfold',FEAT.label0,nFolds);
    FEAT.allFeature0=[];
    for j = 1:num
        FEAT.allFeature0 = [FEAT.allFeature0 FEAT.allFeatures(:,FEAT.rank_all_0(j))];
    end
% FEAT.allFeature0 = [FEAT.allFeatures(:,FEAT.rank_all_0(1)),FEAT.allFeatures(:,FEAT.rank_all_0(2))];
    for fold = 1:nFolds
        test = (indices == fold); train = ~test;    
        % train model on train data
        model.class0 = trainShrinkLDA(FEAT.allFeature0(train,:),FEAT.label0(train),0.1,'');

        y0 = predictShrinkLDA(model.class0,FEAT.allFeature0(test,:));

        MCR_0(fold) = sum(FEAT.label0(test) ~= y0')/length(FEAT.labels(test)); % Ref 

    end
avgMCR_0 = mean(MCR_0); % Ref
lda_e(lda,1) = avgMCR_0;
disp('aver_mean_0_12')
disp(avgMCR_0)


% Cross validation for 1&2
% Build 1 vs 0&2
FEAT.label1 = Label;
FEAT.allFeature_1 = FEAT.allFeatures;
FEAT.allFeature_1(FEAT.label1==0,:) = [];
FEAT.label1(FEAT.label1==0)=[];
FEAT.label1(FEAT.label1==2)=-1;
FEAT.label1(FEAT.label1==1)=1;
[FEAT.d_all_1 FEAT.rank_all_1] = fisherrank(FEAT.allFeature_1,FEAT.label1);


nFolds = 8;
indices = crossvalind('Kfold',FEAT.label1,nFolds);
FEAT.allFeature1=[];
    for j = 1:num
        FEAT.allFeature1 = [FEAT.allFeature1 FEAT.allFeature_1(:,FEAT.rank_all_1(j))];
    end
% FEAT.allFeature1 = [FEAT.allFeature1(:,FEAT.rank_all_1(1)),FEAT.allFeature1(:,FEAT.rank_all_1(2))];
    for fold = 1:nFolds
        test = (indices == fold); train = ~test;    
        % train model on train data
        model.class1 = trainShrinkLDA(FEAT.allFeature1(train,:),FEAT.label1(train),0.1,'');

        y1 = predictShrinkLDA(model.class1,FEAT.allFeature1(test,:));

        MCR_1(fold) = sum(FEAT.label1(test) ~= y1')/length(FEAT.labels(test)); % Ref 

    end
avgMCR_1 = mean(MCR_1); % Ref

disp('aver_mean_1_2')
disp(avgMCR_1)
lda_e(lda,2) = avgMCR_1;
end

lda_error = mean(lda_e);
disp('average error of lda for 2 models')
disp(lda_error)
% 
% if visualize
%     figure(5)
%     plot(avgMCR_it*100)
%     hold on
%     xlabel('number of best features')
%     ylabel('MCR [%]')
%     grid on
% end
error_of_lda(ww,:)=lda_error;
end
bw(:,num)=error_of_lda(:,2);
end
%% Classification/Modelling-----------KNN
FEAT.labels = Label;



%% Validation for KNN
av = zeros(100,1);
for i = 1:100
nFolds = 8;
indices = crossvalind('Kfold',FEAT.labels,nFolds);
for fold = 1:nFolds
    test = (indices == fold); train = ~test;     
    mdl = fitcknn(FEAT.allFeatures(train,:),FEAT.labels(train));
%     p0
%     p1 = predictShrinkLDA(model.class1,FEAT.allFeature1(test,:));
    p2 = predict(mdl,FEAT.allFeatures(test,:));
%     p3 = predict(SVMModel0,FEAT.allFeaturestest(test,:));

%     MCR_p1(fold) = sum(FEAT.labels(test) ~= p1')/length(FEAT.labels(test)); % Ref 
    MCR_p2(fold) = sum(FEAT.labels(test) ~= p2)/length(FEAT.labels(test));
%     MCR_p3(fold) = sum(FEAT.label0(test) ~= p3')/length(FEAT.labels(test));
end
% avgMCR_p1 = mean(MCR_p1);
av(i) = mean(MCR_p2);
% avgMCR_p3 = mean(MCR_p3);
end
disp('average error rate of knn')
disp(mean(av))
% disp(avgMCR_p3)


%% Classification/Modelling-----------SVM

nFolds = 8;
indices = crossvalind('Kfold',FEAT.label1,nFolds);
for fold = 1:nFolds
    test = (indices == fold); train = ~test;     
    SVMModel0 = fitcsvm(FEAT.allFeature1(train,:),FEAT.label1(train));

    p3 = predict(SVMModel0,FEAT.allFeature1(test,:));
 
    MCR_p3(fold) = sum(FEAT.label0(test) ~= p3)/length(FEAT.labels(test));
end
avgMCR_p3 = mean(MCR_p3);
% avgMCR_p3 = mean(MCR_p3);
disp('average error rate of svm')
disp(avgMCR_p3)
%finally Marko is responsible for the intergration of MATLAB and Openvibe