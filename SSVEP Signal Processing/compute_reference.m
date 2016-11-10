function [O1_ref,O2_ref] = compute_reference(filename1,visualization) 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Project Name:SSVEP based BCI in controlling NAO robot
%Author:      Erxin Wang
%Date:        21.July. 2016

%%Parameters
%Input:filename1      Session Reference1
%      filename2      Session Reference2
%Output:O1_ref        Channel O1 reference w.r.t PWelch methods
%       O2_ref        Channel O2 reference
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %% Initialization
    epoch_time = 10;
    epochs = [];
    epoch_start = 0.25;
    epoch_period = 4.5;
    % vis = 2;
    % 1 = O1
    % 2 = O2
    [~,M1] = edfread(filename1);
%     [~,M2] = edfread(filename2);
%     M = [M1(3:16,641:16000),M2(3:16,641:16000)]; %extrach the 14 channels
    M = [M1(3:16,641:16000)];
    trial = 12;
    channel = [7,8];
    %7 = O1
    %8 = O2

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
    % alternative: high-pass filter with cutoff 1 Hz

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

    %% Feature extraction
    %buil the the standard dataset matrix
    for e = 1:trial
        epochs(:,:,e) = M(channel,(epoch_start+epoch_time*(e-1))*128+640+1:...
            (epoch_start+epoch_time*(e-1)+epoch_period)*128+640);
        %cut off the beginning 5 sec
    end
    window = 4*128;
    noverlap = 510;
    nfft = 4*128;
    fs = 128;
    for e = 1:12
        [pxx1(:,e),f1] = pwelch(epochs(1,1:end,e),window,noverlap,nfft,fs);
        [pxx2(:,e),f2] = pwelch(epochs(2,1:end,e),window,noverlap,nfft,fs);
    end
    O1_ref = mean(pxx1(1:end,:),2);
    O2_ref = mean(pxx2(1:end,:),2);

    if visualization
        plot(f1,O1_ref,f2,O2_ref,'LineWidth',2);legend('O1','O2'),grid on
    end
end
%'S1-REF4_1-20.07.16.20.05.14.edf','S1-REF4_2-20.07.16.20.16.32.edf'

