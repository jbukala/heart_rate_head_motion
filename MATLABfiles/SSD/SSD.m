function [SSDcomponents]=SSD(v,Fs,th,maxNumberofComponents)

% Function to decompose a given signal by means of Singular Value Decomposition.

% Refactored and optimized by Fabian Fraenz

% INPUTS:
% v: mono-variate input signal to decompose
% Fs: sampling frequency
% th: threshold which defines the variance of the residual (default 0.01),
% at which the decomposition stops. A threshold of 0.01 means that when the
% residual reaches 1% or less of the variance of the original signal v, the
% algorithm stops decomposing that residual.
% maxNumberofComponents: stops the decomposition after
% maxNumberofComponents have been retrieved

% OUTPUTS:
% SSDcomponents: SSD components of the original signal v. Components are 
% ordered in decreasing order of frequency content.

% EXAMPLE:
% Creation of some simple timeserie
% y  = sin(2*pi*5*(0:999)/1000);
% y2 = 0.1*sin(2*pi*15*(0:999)/1000);
% y3 = y+y2;
% y3(500:999) = y3(500:999)+sin(2*pi*75*(500:999)/1000);
%
% x1 = sin(2*pi*5*(0:999)/1000);
% x2 = [zeros(1,500) sin(2*pi*75*(501:1000)/1000)];
% x3 = 0.1*sin(2*pi*15*(0:999)/1000);
%
% v  = y3;
%
% Sampling frequency 1000 and threshold of 0.005
%
% SSDcomponents = SSD(v,1000,0.005);
%

warning off;

if nargin < 3 || isempty(th)
    th = 0.01;
end

if nargin < 4 || isempty(maxNumberofComponents)
    maxNumberofComponents = 1000;
end

v=v(:)';
L=length(v);

v = v-mean(v);

orig = v;

RR1=zeros(size(v));
k1 = 0;

if Fs/L <= 0.5
    lf = L;
else
    lf = 2*Fs;
end

remen = 1;
testcond = 0;

while (remen > th) && (k1 < maxNumberofComponents)
    k1 = k1+1;
    v = v-mean(v);
    v2 = v;
    
    clear nmmt
    [nmmt,ff] = pwelch(v2,[],[],4096,Fs);
    [~,in3] = max(nmmt);
    nmmt = nmmt';
    
    if ((k1 == 1) && (ff(in3)/Fs < 1e-3)) % trend detection at first iteration
        l = floor(L/3);
        M = zeros(L-(l-1), l);
        for k=1:L-(l-1),
            M(k,:) = v2(k:k+(l-1));
        end
        [U,S,V] = svd(M);
        U(:,l+1:end) = [];
        S(l+1:end,l+1:end) = [];
        V(:,l+1:end) = [];
        
        rM = rot90(U(:,1)*S(1,:)*V');
        r = zeros(1,L);
        [~,m] = size(rM);
        for k=-(l-1):L-(l),
            r(k+l) = sum(diag(rM,k))/m;  
        end
    else % if no trend detected, or after trend removal
        for cont = 1:2
            v2 = v2-mean(v2);
            [deltaf] = gaussfitSSD(ff,nmmt'); % Gaussian fit of spectral components
            % estimation of the bandwidth
            [~,iiii1] = min(abs(ff-(ff(in3)-deltaf)));
            [~,iiii2] = min(abs(ff-(ff(in3)+deltaf)));
            l = floor(Fs/ff(in3)*1.2);
            
            if l <= 2 || l > floor(L/3)
                l = floor(L/3);
            end
            
            M=zeros(L, l);
            % M built with wrap-around
            for k=1:l,
                M(:,k)=[v2(end-k+2:end)'; v2(1:end-k+1)'];
            end
            
            [U,S,V] = svd(M,0);
            
            %% Selection of all principal components with a dominant frequency
            % inside the estimated band-width
            if size(U,2)>l
                yy = abs(fft(U(:,1:l),lf));
            else
                yy = abs(fft(U,lf));
            end
            yy_n = size(yy,1);
            ff2 = (0:yy_n-1)*Fs/yy_n;
            yy(floor(yy_n/2)+1:end,:) = [];
            ff2(floor(length(ff2)/2)+1:end) = [];
            % %
            if size(U,2)>l
                [~,ind1] = max(yy(:,1:l));
            else
                [~,ind1] = max(yy);
            end
            
            ii2 = find(ff2(ind1)>ff(iiii1) & ff2(ind1)<ff(iiii2)); %0.31
            [~,indom] = min(abs(ff2-ff(in3)));
            [~, maxindom] = max(yy(indom,:));

            if isempty(ii2)
                rM=U(:,1)*S(1,:)*V';
            else
                if ii2(ii2==maxindom)
                    rM = U(:,ii2)*S(ii2,:)*V';
                else
                    ii2 = [maxindom,ii2];
                    rM = U(:,ii2)*S(ii2,:)*V';
                end
            end
            
            %%
            if cont == 2
                vr = r;
            end
           
            [~,m] = size(rM);
 
            for k=-(L-1):0,
                kl = k+L;
                if kl >= m
                    r(kl) = sum(diag(rM,k))/m;    
                else
                    r(kl) = (sum(diag(rM,k))+sum(diag(rM,kl)))/m;
                end
            end
            r = fliplr(r);
            
            if cont == 2 && r*(v-r)'<0 % check condition for convergence
                r = vr;
            end
            v2 = r;
        end
    end
    RR1(k1,:) = (v*r'/(r*r'))*r;
    v=v-RR1(k1,:);
    remenold = remen;
    if testcond
        remen = sum((sum(RR1(stept:end,:),1)-orig2).^2)/(sum(orig2.^2));
        if k1 == stept+3;
            break
        end
    else
        remen = sum((sum(RR1(1:end,:),1)-orig).^2)/(sum(orig.^2));
    end
    % in rare cases, convergence becomes very slow; the algorithm is then
    % stopped if no real improvement in decomposition is detected (this is 
    % something to fix in future versions of SSD)
    if abs(remenold - remen)< 1e-5 
        testcond = 1;
        stept = k1+1;
        orig2 = v;
    end
end

% Notify the user that the there is to much noise in the signal to
% accurately decompose it.
if testcond
    fprintf('warning: noise level affecting decomposition, total energy described by SSD components is: %3.1f\n %',(1-sum((sum(RR1(1:end,:),1)-orig).^2)/(sum(orig.^2)))*100);
end

ftemp = (0:size(RR1,2)-1)*Fs/size(RR1,2);
sprr = abs(fft(RR1'));
[~,isprr] = max(sprr);
fsprr = ftemp(isprr);
[~,iord] = sort(fsprr,'descend');
RR1 = RR1(iord,:);

SSDcomponents = RR1;
