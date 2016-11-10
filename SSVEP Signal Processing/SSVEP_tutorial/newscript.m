clear all
close all

[hdr1 data1] = edfread('S1-ref_eyesclosed-01.07.16.17.28.23.edf');
[hdr2 data2] = edfread('S1-ref_eyesopen-01.07.16.17.27.19.edf');
[hdr3 data3] = edfread('S1-stim7Hz-01.07.16.17.29.29.edf');
[hdr4 data4] = edfread('S1-stim7Hzb-01.07.16.18.01.53.edf');

ch = 9;
x1 = data1(ch,:);
x2 = data2(ch,:);
x3 = data3(ch,:);
x4 = data4(ch,:);

[fx1 fpara] = bandfilter(x1,3,35,128);
[fx2 fpara] = bandfilter(x2,3,35,128);
[fx3 fpara] = bandfilter(x3,3,35,128);
[fx4 fpara] = bandfilter(x4,3,35,128);

% fx1 = x1-mean(x1);
% fx2 = x2-mean(x2);
% fx3 = x3-mean(x3);
% fx4 = x4-mean(x4);

figure(1)
plot(fx1)
hold on
plot(fx2,'g')
plot(fx3,'r')
plot(fx4,'m')
legend('eyes open','eyes closed','7Hz stim','7Hz with 1 LEDs')

window = 2048;
noverlap = 2000;
nfft = 2048;
fs = 128;
[pxx1,f] = pwelch(fx1,window,noverlap,nfft,fs);
[pxx2,f] = pwelch(fx2,window,noverlap,nfft,fs);
[pxx3,f] = pwelch(fx3,window,noverlap,nfft,fs);
[pxx4,f] = pwelch(fx4,window,noverlap,nfft,fs);

figure(2)
plot(f,pxx1)
hold on
plot(f,pxx2,'g')
plot(f,pxx3,'r')
plot(f,pxx4,'m')
legend('eyes open','eyes closed','7Hz stim','7Hz with 1 LEDs')

[s,f,t,p1] = spectrogram(fx1,window,noverlap,nfft,fs);

[s,f,t,p2] = spectrogram(fx2,window,noverlap,nfft,fs);

[s,f,t,p3] = spectrogram(fx3,window,noverlap,nfft,fs);

figure(3)
subplot(2,2,1)
imagesc(p1,[0 150])
subplot(2,2,2)
imagesc(p2,[0 150])
subplot(2,2,3)
imagesc(p3,[0 150])

l = 2;
h = 30;
[ffx1 fpara] = bandfilter(x1,l,h,128);
[ffx2 fpara] = bandfilter(x2,l,h,128);
[ffx3 fpara] = bandfilter(x3,l,h,128);
[ffx4 fpara] = bandfilter(x4,l,h,128);

figure(4)

%plot(ffx2.^2,'g')
plot(ffx1.^2,'b')
plot(ffx3.^2,'r')
hold on
plot(ffx4.^2,'m')

legend('eyes open','7Hz stim','7Hz with 1 LEDs')

rms_ref = sqrt(mean(ffx1.^2,2))
rms_stim1 = sqrt(mean(ffx3.^2,2))
rms_stim2 = sqrt(mean(ffx4.^2,2))