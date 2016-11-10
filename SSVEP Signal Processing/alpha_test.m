clear all
close all
[hdr,data] = edfread('S4-2REF-19.07.16.22.42.47.edf');
ch = 9;
x2 = data(ch,:);
[fx2,~] = bandfilter(x2,3,50,128);
figure(1)
plot(fx2)
window = 1500;
noverlap = 1400;
nfft = 1500;
fs = 128;
[pxx2,f] = pwelch(fx2,window,noverlap,nfft,fs);
figure(2)
plot(f,pxx2,'g')