%plot cepstrum:
clear all
Fs = 30;
num = csvread('..\output\PCA_Signals.csv');

x = num(9,:);
y = fft(x);
y = abs(y);
z = log(y);
cepstrum = ifft(z);

plot(x);
hold on;
plot(y, 'r');
figure,
plot(cepstrum,'k');