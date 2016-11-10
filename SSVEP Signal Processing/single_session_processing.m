clear all
close all

%% Initialization
epoch_time = 10;
epochs = [];
epoch_start = 0.25;
epoch_period = 4.5;
[O1,O2] = compute_reference('S1-REF4_1-20.07.16.20.05.14.edf','S1-REF4_2-20.07.16.20.16.32.edf',1);
filename = 'S1-D4-20.07.16.20.13.10.edf';
[hdr1,M] = edfread(filename);
M = M(3:16,:); %extrach the 14 channels
len = size(M,2);
trial = 12;
t = linspace(0,len/128,len); %generate the time line
% lab = repmat([1 2],1,10/2);% generate the lab vector
lab = [1 2 2 1 1 1 2 2 1 2 2 1]; %session1
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

%% Preprocessing
% spatial filtering 
CAR = mean(M);
for ch = 1:14
    M(ch,:) = M(ch,:)-CAR;
end
% detrending
for ch = 1:14
    M(ch,:) = M(ch,:)-mean(M(ch,:),2);
end
% Alternative: high-pass filter with cutoff 1 Hz

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

for e = 1:trial
    epochs(:,:,e) = M(channel,(epoch_start+epoch_time*(e-1))*128+640+1:...
        (epoch_start+epoch_time*(e-1)+epoch_period)*128+640);
%     fepochs(:,:,e) = fx(:,epoch_time*(e-1)*128+1:(epoch_time*e)*128);
end

% s1 = epochs(:,:,lab==1);%7.5Hz
% s2 = epochs(:,:,lab==2);%10 Hz
% 
% stim_rms = sqrt(mean(fepochs(1,:,lab==1).^2,2))
% ref_rms = sqrt(mean(fepochs(1,:,lab==2).^2,2))
% 
% boxplot(squeeze(stim_rms))
% figure(3)
% plot(squeeze(s1(1,:,1)))
% hold on
% plot(squeeze(s2(1,:,1)),'g')
% title('Bandfiltered signals of the first trial');
% legend('7.5Hz','10Hz')

window = 4*128;
noverlap = 510;
nfft = 4*128;
fs = 128;
for e = 1:12
    [pxx1(:,e),f1] = pwelch(epochs(1,1:end,e),window,noverlap,nfft,fs);
    [pxx2(:,e),f2] = pwelch(epochs(2,1:end,e),window,noverlap,nfft,fs);
end

figure(3)
subplot(4,2,1)
plot(f1(1:end),pxx1(1:end,lab==1))
hold on
plot(f1(1:end),mean(pxx1(1:end,lab==1),2),'k','LineWidth',3)
title('The 7.5Hz signal O1')
legend('Trial1','Trial2','Trial3','Trial4','Trial5','Trial6','Average')
axis([0,40,-inf,inf])

subplot(4,2,3)
plot(f1(1:end),pxx1(1:end,lab==2))
hold on
plot(f1(1:end),mean(pxx1(1:end,lab==2),2),'k','LineWidth',3)
title('The 12Hz signal O1')
legend('Trial1','Trial2','Trial3','Trial4','Trial5','Trial6','Average')
axis([0,40,-inf,inf])

subplot(4,2,5)
plot(f1(1:end),mean(pxx1(1:end,lab==1),2),'b','LineWidth',3)
hold on
plot(f1(1:end),mean(pxx1(1:end,lab==2),2),'g','LineWidth',3)
title('The averaged 7.5Hz signal and 12Hz signal O1');xlabel('Frequency(Hz)');
legend('7.5Hz','12Hz')
axis([0,40,-inf,inf])

subplot(4,2,7)
plot(f1(1:end),mean(pxx1(1:end,lab==1),2)-O1,'b','LineWidth',3)
hold on
plot(f1(1:end),mean(pxx1(1:end,lab==2),2)-O1,'g','LineWidth',3)
title('The baseline removed 7.5Hz signal and 12Hz signal O1');xlabel('Frequency(Hz)');
legend('7.5Hz','12Hz')
axis([0,40,-inf,inf])

subplot(4,2,2)
plot(f2(1:end),pxx2(1:end,lab==1))
hold on
plot(f2(1:end),mean(pxx2(1:end,lab==1),2),'k','LineWidth',3)
title('The 7.5Hz signal O2')
legend('Trial1','Trial2','Trial3','Trial4','Trial5','Trial6','Average')
axis([0,40,-inf,inf])

subplot(4,2,4)
plot(f2(1:end),pxx2(1:end,lab==2))
hold on
plot(f2(1:end),mean(pxx2(1:end,lab==2),2),'k','LineWidth',3)
title('The 12Hz signal O2')
legend('Trial1','Trial2','Trial3','Trial4','Trial5','Trial6','Average')
axis([0,40,-inf,inf])

subplot(4,2,6)
plot(f2(1:end),mean(pxx2(1:end,lab==1),2),'b','LineWidth',3)
hold on
plot(f2(1:end),mean(pxx2(1:end,lab==2),2),'g','LineWidth',3)
title('The averaged 7.5Hz signal and 12Hz signal O2');xlabel('Frequency(Hz)');
legend('7.5Hz','12Hz')
axis([0,40,-inf,inf])

subplot(4,2,8)
plot(f2(1:end),mean(pxx2(1:end,lab==1),2)-O2,'b','LineWidth',3)
hold on
plot(f2(1:end),mean(pxx2(1:end,lab==2),2)-O2,'g','LineWidth',3)
title('The baseline removed 7.5Hz signal and 12Hz signal O2');xlabel('Frequency(Hz)');
legend('7.5Hz','12Hz')
axis([0,40,-inf,inf])

% figure(4)
% spectrogram(epochs(1,1:end,5),window,noverlap,nfft,fs,'yaxis')
% % axis([10 20 2 3 -inf 10])
% view([90,0])


