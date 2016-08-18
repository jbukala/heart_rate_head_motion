%Signal decomposition using multivariate Singular Spectrum Decomposition (SSD).
%mainly used for some first testing.

%Add SSD-folder to path for this to work

%num = csvread('..\output\clustered_signals.csv');
%num = csvread('..\output\PCA_Signals.csv');
num = csvread('..\output\rawSignals.csv');

v = num(1,:);

maxComponents = 5;
th=0.01;
[SSDcomps] = SSD( v, Fs, th, maxComponents);

figure,
plot(SSDcomps(1,:))
hold on
plot(SSDcomps(2,:), 'r')
hold on
plot(SSDcomps(3,:), 'k')

%use FFT to check spectrum and max freq:
sig = SSDcomps(2,:);
L = length(sig);
Y = fft(sig);
P2 = abs(Y/L);
P1 = P2(1:floor(L/2)+1);
[~, I] = max(P1);
f = Fs*(0:floor(L/2))/L;

FFT_HR = f(I) * 60;
figure,
plot(f, P1)

figure,
z = hilbert(SSDcomps(3,:));
instfreq = Fs/(2*pi)*diff(unwrap(angle(z)));
plot(instfreq)

figure,
spectrogram(SSDcomps(3,:),200,180,256,Fs,'yaxis')
ax = gca;
shading interp

%csvwrite('..\output\SSD_Signals.csv', SSDcomps);