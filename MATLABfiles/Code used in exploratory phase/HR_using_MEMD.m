%Signal decomposition using multivariate Empirical Mode Decomposition.
%mainly used for some first testing.

%Add MEMD-folder to path for this to work

% num = csvread('..\output\clustered_signals.csv');
%num = csvread('..\output\PCA_Signals.csv');

num = csvread('..\output\rawSignals.csv');

signals = num(1:16,:)';
MEMD_result = memd(signals);

imf_x = reshape(MEMD_result(1,:,:),size(MEMD_result,2),length(signals)); % imfs corresponding to 1st component
imf_y = reshape(MEMD_result(2,:,:),size(MEMD_result,2),length(signals)); % imfs corresponding to 2nd component
imf_z = reshape(MEMD_result(3,:,:),size(MEMD_result,2),length(signals)); % imfs corresponding to 3rd component

figure,
plot(imf_z(1,:))
hold on
plot(imf_z(2,:), 'r')
hold on
plot(imf_z(3,:), 'k')
%hold on
%plot(imf_y(4,:), 'c')

figure,
z = hilbert(imf_x(1,:));
instfreq = Fs/(2*pi)*diff(unwrap(angle(z)));
plot(instfreq)

figure,
spectrogram(imf_x(1,:),200,180,256,Fs,'yaxis')
ax = gca;
shading interp

%use FFT to check spectrum and max freq:
sig = imf_z(2,:);
L = length(sig);
Y = fft(sig);
P2 = abs(Y/L);
P1 = P2(1:floor(L/2)+1);
[~, I] = max(P1);
f = Fs*(0:floor(L/2))/L;

FFT_HR = f(I) * 60;
figure,
plot(f, P1)

%csvwrite('..\output\MEMD_Signals.csv', MEMD_result);

