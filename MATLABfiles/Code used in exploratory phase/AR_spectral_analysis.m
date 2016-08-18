%% Exercise 2.1
data = csvread('..\output\PCA_Signals.csv');
x = data(1,:);
x = x - mean(x);
Fs = 30; %sampling freq
n = 9; %order of the AR model
%make zero mean
% calculate autocorrelation
R=zeros(size(x));
N=length(x);
for cnt=1:length(x)
    for k=1:N-(cnt-1)
        R(cnt)=R(cnt)+x(k)*x(k+(cnt-1));
    end
    R(cnt)=R(cnt)/N;
end
%autocorrelation for positive lags
r=R(2:end);
% autocorrelation at lag 0
r0=R(1);

%% Exercise 2.3
% Levinson algorithm
A=zeros(n,n);
sigma2=zeros(1,n);
A(1,1)=-1*r(1)/r0;
sigma2(1)=(1-A(1,1)^2)*r0;
for k=2:n
    A(k,k)=r(k);
    for i=1:k-1
        A(k,k)=A(k,k)+A(k-1,i)*r(k-i);
    end
    A(k,k)=-1*A(k,k)/sigma2(k-1);
    for i=1:k-1
        A(k,i)=A(k-1,i)+A(k,k)*A(k-1,k-i);
        sigma2(k)=(1-A(k,k)^2)*sigma2(k-1);
    end
end

% alternative algorithm

% b=zeros(n,n);
% v=zeros(1,n);
% b(1,1)=r(1)/r0;
% v(1)=r0*(1-b(1,1)^2);
% for k=2:n
%     b(k,k)=(r(k)-b(k-1,1:k-1)*(r(k-1:-1:1))')/v(k-1);
%     b(k,1:k-1)=b(k-1,1:k-1)-b(k,k)*b(k-1,k-1:-1:1);
%     v(k)=v(k-1)*(1-b(k,k)^2);
% end

% % Error filter
% out=filter([1 A(end,:)],1,ClippingReconstructed-mean(ClippingReconstructed));
% % Prediction filter (with a time delay)
% pred=filter(-1*A(end,:),1,ClippingReconstructed-mean(ClippingReconstructed))+mean(ClippingReconstructed);
% plot(out(1:end-1));hold on;plot(pred(1:end-1),'g-');plot(ClippingReconstructed(2:end),'r:');
% legend({'prediction error','prediction','signal'});

% sys1=tf(1,[1 A(end,:)],1/1000);
% noisedriving=sigma2(end)*randn(5000,1);
% lsim(sys1,noisedriving',(1:length(noisedriving))/1000);

 
%% Exercise 2.4
% AR PSD
clear i
T=1/Fs;
omega=linspace(0,2*pi,length(x));
omega(end)=[];
ARPSD=zeros(size(omega));
for cnt=1:length(omega)
    tmp=1;
    for k=1:n
        tmp=tmp+A(n,k)*exp(-i*omega(cnt)*k);
    end
    ARPSD(cnt)=sigma2(n)/abs(tmp)^2;
end
figure;
plot(Fs*omega/(2*pi),ARPSD)
hold on;
X=fft(x);
w=linspace(0,Fs,length(x)+1);
w(end)=[];
plot(w,(1/length(x))*(X.*conj(X)),'r--')
legend({'Auto Regressive Power Spectral Density','Spectrogram PSD estimation'})
xlabel('frequency Hz')
