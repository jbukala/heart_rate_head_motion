M = csvread('..\output\PCA_Signals.csv');

fs = 30;
peakheights = zeros(10,1);

sig = M(2,:);
[autocor,lags] = xcorr(sig,2*fs,'coeff'); 
[pksh,lcsh] = findpeaks(autocor);
peak = pksh(round(size(lcsh,1)/2));

plot(lags/fs,autocor, 'b')%this is the best signal
xlabel('Lag (secs)')
ylabel('Autocorrelation')
hold on;

for i=1:10
sig = M(i,:);
[autocor,lags] = xcorr(sig,2*fs,'coeff');
%plot(lags/fs,autocor, 'r')
[pksh,lcsh] = findpeaks(autocor);

peakheights(i) = pksh(round(size(lcsh,1)/2)); 

%if pksh(round(size(lcsh,1)/2)) > peak
%    i
%end
end;

peakheights
