Fs = 61;
num = csvread('..\output\PCA_Signals.csv');



Y = fft(sig);
P2 = abs(Y/L);
P1 = P2(1:floor(L/2)+1);
[~, I] = max(P1);
f = Fs*(0:floor(L/2))/L;

FFT_HR = f(I) * 60;