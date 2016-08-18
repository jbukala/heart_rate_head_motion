%Implementation of signal selection using DCT from:
%"Improved Pulse Detection from Head Motions Using DCT" - paper
%Some steps changed in an attempt to retrieve better results

%Fs = 61;
num = csvread('..\output\PCA_Signals.csv');

Q = zeros(size(num,1),1);
minKh = zeros(size(num,1),1);
SCminKh = zeros(size(num,1),1);
L = size(num,2);

for i=1:1%size(num,1) %for each signal S_i
    S = num(i,:) - mean(num(i,:));
    %S = num(i,:));
    pow = sum(S.^2);
    SC = dct(S); %calc SC as the DCT of S
    
    K = [];
    tmpSC = abs(SC);
    while sum(SC(K).^2)/pow < 0.50 % K is smallest indices of SC s.t. 50% of the power of signal S is contained in them
        [M, I] = max(tmpSC);
        tmpSC(I) = 0;
        K(length(K)+1) = I;
    end
    
    K = sort(K);
    Kh_len = min(5,length(K));
    Kh = K(1:Kh_len);
    Kh = Kh(Kh<floor(length(SC)/2)); %remove all elements where 2*Kh is outside SC
    
    Q(i) = norm([SC(Kh) SC(2*Kh)])/norm(SC);
    
    %[~, maxcoeff] = max(abs(SC(Kh)));
    %minKh(i) = Kh(maxcoeff); %use largest power coeff from Kh, might be smarter?
    [~, minpowcoeff] = max(SC(Kh).^2); %min(SC(Kh).^2); %use min power coeff of Kh
    minKh(i) = Kh(minpowcoeff); 
    
    %minKh(i) = min(Kh); %<-papers definition
    SCminKh(i) = SC(minKh(i));
end

[~, HRsig] = max(Q); %signal with max Q is the HR-signal

tmpSig = zeros(1,size(num,2));
tmpSig(minKh(HRsig)) = SCminKh(HRsig);
oneCompSig = idct(tmpSig);
Y = fft(oneCompSig);
P2 = abs(Y/L);
P1 = P2(1:floor(L/2)+1);
[~, I] = max(P1);
f = Fs*(0:floor(L/2))/L;

DCT_HR = f(I) * 60;