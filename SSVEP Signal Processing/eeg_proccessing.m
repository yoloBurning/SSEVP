clear all
close all
clc
filename = 'E1_10.csv';
fileID = fopen(filename);
M_head = textscan(fileID,'%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s'...
    ,1,'HeaderLines',0,'Delimiter',',');
M_head(1) = [];
M_head(end) = [];
fclose(fileID);
M = csvread(filename,1);%read from the 2 row
M = M(:,2:end-1); %delete the irrelevant columns, the first is time the last is sampling rate
% L = size(M,1);
x = 0.5*(M(:,7))+0.5*(M(:,8));% 7 stands for the O1 channel and  8 for O2 channel
x_ssvep = 0; % signal by frequency evoking
x_break = 0; % signal at rest
for i = 1
    x_ssvep = x_ssvep + x(( 6.5+10*(i-1))*128+1: (6.5+10*(i-1)+3)*128); %select epoch
    x_break = x_break + x((11.5+10*(i-1))*128+1:(11.5+10*(i-1)+3)*128);
end
% x_ssvep = x_ssvep/10;
% x_break = x_break/10;
L = length(x_ssvep);

%% Exploration of the raw signal
figure(1);
subplot(2,1,1);plot(x_ssvep,'r');xlabel('ssvep sample');title('ssvep raw signal')
grid on;
subplot(2,1,2);plot(x_break);xlabel('break sample');title('break raw signal')
grid on;

%% Extraction of statistical measures 
% subplot(2,1,2)
% hist(x);
% grid on;
% x_mean = mean(x);
% x_median = median(x);
% x_std = std(x);
% x_var = x_std.^2;
% x_pow = sum(x.^2)/length(x);
% x_RMS = sqrt(x_pow);
% fprintf('mean is %f\nmedian is %f\nstandard deviation is %f\nvariance is %f\npower is %f\nRMS is %f\n',x_mean,x_median,x_std,x_var,x_pow,x_RMS);

%% Detrending the signal
%Remove the offset of the signal
x_ssvep = detrend(x_ssvep);
x_break = detrend(x_break);
% figure(2)
% plot(x,'b')
% hold on
% plot(x_det,'g')
% hold off
% legend('row data','detrended data')
% xlabel('sample')

%% Frequency decomposition using FFT
%-----------time domain---------------------
Fs = 128;
T = 1/Fs;
t = (0:L-1)*T; % Time vector

figure(2);
subplot(2,1,1);plot(t,x_ssvep,'r');xlabel('time/s');title('ssvep raw signal');
grid on;
subplot(2,1,2);plot(t,x_break);xlabel('time/s');title('break raw signal')
grid on;

%---------frequency domain------------------
a = L;
X_ssvep = fft(x_ssvep,a);
P2_ssvep = abs(X_ssvep/a); %nomilized amplitude
P1_ssvep = P2_ssvep(1:a/2+1); %single-side amplitude spectrum
P1_ssvep(2:end-1) = 2*P1_ssvep(2:end-1);
f_ssvep = Fs*(0:(a/2))/a;
X_break = fft(x_break,a);
P2_break = abs(X_break/a); %nomilized amplitude
P1_break = P2_break(1:a/2+1); %single-side amplitude spectrum
P1_break(2:end-1) = 2*P1_break(2:end-1);
f_break = Fs*(0:(a/2))/a;
figure(3)
subplot(2,1,1)
plot(f_ssvep,P1_ssvep,'r');
% axis([0 30 0 3]);
title('Single-Sided Amplitude Spectrum of X ssvep(t)')
xlabel('f(Hz)');ylabel('|X(f)|');grid on;
subplot(2,1,2)
plot(f_break,P1_break);
% axis([0 30 0 3])
title('Single-Sided Amplitude Spectrum of X break(t)')
xlabel('f(Hz)');ylabel('|X(f)|');grid on;


%% Moving average filtering


