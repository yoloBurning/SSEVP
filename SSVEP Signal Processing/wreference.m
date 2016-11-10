clear all
close all

%% Initialization
epoch_time = 10;
% vis = 2;
% 1 = O1
% 2 = O2

% filename = 'S1-stim_14Hz_10seconds_5trials-05.07.16.18.11.25.edf';
% filename = 'S1-stim_6Hz_10sec_epoching_5trials-05.07.16.18.06.10.edf';
% filename = 'S2-stim_6Hz_10secondsepoch_5trials-05.07.16.19.09.55.edf';
filename = 'S1-3REF-20.07.16.14.48.36.edf';
[hdr1,M] = edfread(filename);
M = M(3:16,:); %extrach the 14 channels
len = size(M,2);
trial = 12;
t = linspace(0,len/128,len); %generate the time line
channel = [7,8];
%7 = O1
%8 = O2

figure(1)
plot(t,M(channel(1),:),t,M(channel(2),:)+100)
legend(hdr1.label{channel(1)+2},hdr1.label{channel(2)+2})
grid on
title('Row Signal');xlabel('Time(seconds)')
hold on
for i = 0:trial-1
    line([10*i+5,10*i+5],[4000,4400],'Color','red','LineWidth',0.5)
    line([10*i+10,10*i+10],[4000,4400],'Color','green','LineStyle','--','LineWidth',0.5)
end
%visulization of segmentation
hold off

% trial = floor((L/(128*epoch_time))); %generate the trials 

%% Preprocessing
% % detrending
for ch = 1:14
    M(ch,:) = M(ch,:)-mean(M(ch,:),2);
end
% %%Alternative: high-pass filter with cutoff 1 Hz

% band pass filter
low  = 3;
high = 50;
for ch = 1:14
    M(ch,:) = bandfilter(M(ch,:),low,high,128);
end

% artifacts removal
% M = M - ones(14,1)*mean(M);
% [W,~,S] = amuse(M);
% figure(2)
% for i =1:14
%     plot(t,S(i,:)+(i-1)*10);
%     hold on
% end
% S([1,14],:) = [];
% W([1,14],:) = [];
% M = pinv(W)*S; % reconstruct the signal ==> artifacts removal

figure(2)
plot(t,M(channel(1),:),t,M(channel(2),:)+100)
legend(hdr1.label{channel(1)+2},hdr1.label{channel(2)+2})
grid on
title('Artifacts Removed Signal');xlabel('Time(seconds)')
hold on
for i = 0:trial-1
    line([10*i+5,10*i+5],[-40,140],'Color','red','LineWidth',0.5)
    line([10*i+10,10*i+10],[-40,140],'Color','green','LineStyle','--','LineWidth',0.5)
end
%visulization of segmentation
hold off



%% Feature extraction
epochs = [];
epoch_start = 0.5;
epoch_period = 4.5;
for e = 1:trial
    epochs(:,:,e) = M(channel,(epoch_start+epoch_time*(e-1))*128+640+1:...
        (epoch_start+epoch_time*(e-1)+epoch_period)*128+640);
end
window = 4*128;
noverlap = 510;
nfft = 4*128;
fs = 128;
for e = 1:12
    [pxx1(:,e),f1] = pwelch(epochs(1,1:end,e),window,noverlap,nfft,fs);
    [pxx2(:,e),f2] = pwelch(epochs(2,1:end,e),window,noverlap,nfft,fs);
end

figure(3)
subplot(2,1,1)
plot(f1(1:end),pxx1(1:end,:))
hold on
plot(f1(1:end),mean(pxx1(1:end,:),2),'k','LineWidth',3)
title('Reference O1')
axis([0,40,-inf,inf])
subplot(2,1,2)
plot(f1(1:end),pxx2(1:end,:))
hold on
plot(f2(1:end),mean(pxx2(1:end,:),2),'k','LineWidth',3)
title('Reference O2')
axis([0,40,-inf,inf])



