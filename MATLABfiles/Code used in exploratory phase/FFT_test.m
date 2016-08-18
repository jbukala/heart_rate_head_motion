M = csvread('..\output\discardSignals.csv');

%M = csvread('C:\Users\i6093682\Desktop\filtSignal.csv');

Fs = 61;
sigNum = 19;
M = M(:,1:end-1);

%M = M + 100;
% sizeM = size(M);
% L = sizeM(2);
% Y = fft(M(sigNum,:));
% P2 = abs(Y/L);
% P1 = P2(1:L/2+1);
% P1(2:end-1) = 2*P1(2:end-1);
% f = Fs*(0:(L/2))/L;
%plot(f,P1)
%title('Single-Sided Amplitude Spectrum of X(t)')
%xlabel('f (Hz)')
%ylabel('|P1(f)|')

[B, A] = butter(5, (2/Fs).*[0.75 5], 'bandpass');
filtered = filter(B,A,M(sigNum,:));
plot(M(sigNum,:));
hold on;
plot(filtered, 'r');