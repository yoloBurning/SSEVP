%****************************************************************************************************
%
%   2-stage zero-phase cheby2-bandfilter, using first high-pass, then low-pass
%   - Apass defines maximal passband attenuation in dB
%   - Astop defines minimal stopband attenuation in dB
%
%   Author: Stefan Ehrlich
%   Last revised: 12.12.2014
%
%   Input:
%   - xdata: raw or filtered signal segment
%   - low: lower cutoff frequency
%   - high: higher cutoff frequency
%   - srate: sampling rate
%   - 'show': plot frequency response(s) (Default: 0)
%   Output:
%   - fdata: filtered signal
%
%****************************************************************************************************

function [fdata fpara] = bandfilter(xdata,low,high,srate,varargin)

showresponse= 0;

if ~isempty(varargin)
if ischar(varargin{1})
   switch varargin{1}
      case 'show' % nFold to create empirical scores
                showresponse=1; % default = 0;
      end
end
end


Apass = 3;
Astop = 50;
xdata = double(xdata);

%% 1. Highpass
if ~isempty(low)
    Fpass = low;
    %Fstop = low/10;
    Fstop = low-round(low/5);
    Nfreq = srate/2;
    [N1,Ws1] = cheb2ord(Fpass/Nfreq,Fstop/Nfreq,Apass,Astop);
    [B1,A1] = cheby2(N1,Astop,Ws1,'high');
    fdata=filtfilt(B1,A1,xdata);
    if showresponse
        figure
        freqz(B1,A1)
        title('highpass')
    end    
end
%% 2. Lowpass
if ~isempty(high)
    Fpass = high;
    %Fstop = high+high/10*5;
    Fstop = high+round(high/5);
    [N2,Ws2] = cheb2ord(Fpass/Nfreq,Fstop/Nfreq,Apass,Astop);
    [B2,A2] =cheby2(N2,Astop,Ws2,'low');
    fdata=filtfilt(B2,A2,fdata);
    if showresponse
        figure
        freqz(B2,A2)
        title('lowpass')
    end    
    
end

fpara.A1 = A1;
fpara.B1 = B1;
fpara.A2 = A2;
fpara.B2 = B2;

end