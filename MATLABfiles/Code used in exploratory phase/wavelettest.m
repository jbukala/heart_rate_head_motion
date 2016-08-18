Fs = 30;
num = csvread('..\output\PCA_Signals.csv');
signal = num(5,:);
plot(signal);
figure,

lev   = 3;
wname = 'db2';
nbcol = 64;
[c,l] = wavedec(signal,lev,wname);

%plot wavelet coeffs
len = length(signal);
cfd = zeros(lev,len);
for k = 1:lev
    d = detcoef(c,l,k);
    d = d(:)';
    d = d(ones(1,2^k),:);
    cfd(k,:) = wkeep1(d(:)',len);
end
cfd =  cfd(:);
I = find(abs(cfd)<sqrt(eps));
cfd(I) = zeros(size(I));
cfd = reshape(cfd,lev,len);
cfd = wcodemat(cfd,nbcol,'row');

colormap(pink(nbcol));
image(cfd);
tics = 1:lev;
labs = int2str((1:lev)');
ax = gca;
ax.YTickLabelMode = 'manual';
ax.YDir = 'normal';
ax.Box = 'On';
ax.YTick = tics;
ax.YTickLabel = labs;
title('Discrete Wavelet Transform, Absolute Coefficients.');
xlabel('Time (or Space)');
ylabel('Level');