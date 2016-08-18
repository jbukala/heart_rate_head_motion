function HR_from_raw_sig(sigFile, Fs)
%(OPTIONAL: cut signal into 1-sec pieces, discard the pieces with std > 0.215, paste
%signal back together,) then filter, cut signal into 10-sec pieces and use Self-Organizing Maps + k-means clustering to throw away noisy
%points. End up with PCA and save this to a file

%Fs = 61; %sampling rate


%data = csvread('..\output\rawSignals.csv');
data = csvread(sigFile);

%data = data(1:100,:);%to only check GFT points, discard rounded-down landmark points..
%data = data(101:end, :);% to only check landmark points, discard GFT points..
%data = data(:,3*Fs:end); %discard first 3 secs to settle tracker points.

if (size(data,2) > 2135)
    data = data(:,306:2135); %same frames as in DCT+landmark paper and colour-paper
else
    data = data(:,max(1,size(data,2)-(2135-306)):end); %if video is too short otherwise
end

% %Cut signals into pieces:
% T = 1; %timelength in seconds of piece
% num = zeros(size(data,1), size(data,2));
% minp = inf; %shortest signal after cutting out noisy bits
% stdThresh = 0.215;
% 
%     for i=1:size(data,1) %For every point, for every segment check to see if std < 0.215, if so, accept into final sig
%         p = 1;
%         for j=1:floor(size(data,2)/(T*Fs))
%             if std(data(i,1+(j-1)*(T*Fs):j*T*Fs)) < stdThresh
%                 num(i,1+(p-1)*(T*Fs):p*T*Fs) = data(i,1+(j-1)*(T*Fs):j*T*Fs) + (num(i,1+(p-1)*(T*Fs))-data(i,1+(j-1)*(T*Fs))); %paste signals together
%                 %disp('in')
%                 p=p+1;
%             else
%                 %disp('out')
%             end
%         end
%         minp = min(minp, p);
%     end
%     
%     if (minp < 5 || minp == inf)
%         disp('Low quality signal..')
%         minp = 20;
%         num = data;
%     end
%     
%     %cut off the zeros part:
%     num = num(:,1:minp*T*Fs);
%     %minp
  
    num=data;

    %start filtering leftover signal:
    MA_window = 7;
    c = 1;
    d = (1/MA_window) * ones(1, MA_window);
    [b,a] = butter(5, 2/Fs * [0.75 5], 'bandpass'); %0.75 - 5 Hz fifth order butterworth bandpass filter
    
    for i=1:size(num,1) %filter every row
       num(i,:) = filter(d, c, num(i,:)); %First MA-filter
       num(i,:) = filter(b, a, num(i,:)); %Then butterworth bandpass
    end
    num = num(:,2*Fs:end); %throw away first 2 secs, filter artifacts..
    
%     %CUT SIG UP INTO 10-SEC PIECES
%     T = 10; %timelength in seconds of piece
%     data = num;
%     num = zeros(ceil(size(data,1)*size(data,2)/(T*Fs)), T*Fs);
%     colptr = 1;
%     rowptr = 1;
%     piece = 1; %how many segments are there
%     
%     while(colptr < size(data,1))
%         while(size(data,2) - rowptr >= (T*Fs))
%             num(piece,:) = data(colptr,rowptr:(rowptr+(T*Fs)-1)) - data(colptr,rowptr); 
%             rowptr = rowptr+T*Fs;
%             piece = piece+1;
%         end
%         colptr = colptr+1;
%         rowptr = 1;
%     end
%     num = num(1:piece-1,:); %grab the filled part of the matrix as data
%     %END OF CUTTING UP SIGNAL INTO 10-SEC PIECES
    
    %Calculate signals entropy:
    entro = zeros(size(num,1), 1);
    for i=1:size(num,1)
       entro(i) = entropy(num(i,:)); 
    end
    
%     %calculate signals max frequency content: %BAD FEATURE
%     maxFreq = zeros(size(num,1), 1);
%     for i=1:size(num,1)
%         sizeM = size(num);
%         L = sizeM(2);
%         Y = fft(num(i,:));
%         P2 = abs(Y/L);
%         P1 = P2(1:floor(L/2)+1);
%         P1(2:end-1) = 2*P1(2:end-1);
%         f = Fs*(0:floor(L/2))/L;
%         [~, max_idx] = max(P1(2:end-1));
%         maxFreq(i) = f(max_idx+1);
%     end
    
    %Start Self-Organizing Map part:
    x = [abs(mean(num, 2)) std(num, 0, 2) entro]'; %maxFreq might need to go back in there? depending on cluster selection criteria
    %x = [std(num, 0, 2) entro]';
    net = selforgmap([10 10]);
    %net.trainParam.showWindow = false;
    net.trainParam.showCommandLine = false; 
    net = train(net,x);
    y = net(x);
    classes = vec2ind(y)';
    weights = net.IW{1,1};
    
    k=3;
    [idx,C] = kmeans(weights, k);
    
    quality = zeros(k,1);

    for i=1:k
%         minClasses = (idx==i); %SOM-classes which are in the selected k-means cluster
%         used = (minClasses(classes)==1); %signal-pieces
%         quality(i) = sum(used); %Selection criteria to MAXIMIZE to choose
        %which cluster contains the clean signals (select biggest cluster)
        quality(i) = -norm(C(i,:)); %Selection criteria to MAXIMIZE to choose which cluster contains the clean signals (smallest-norm cluster)
    end
    
    [~, minIdx] = max(quality); %now minIdx is the cluster with a center of smallest norm

minClasses = (idx==minIdx); %SOM-classes which are in the selected k-means cluster
used = (minClasses(classes)==1); %signal-pieces to use
usedSignals = num(used,:);

%csvwrite('..\output\SOM_signals.csv', usedSignals);

%perform PCA:
[coeffs, score, ~] = pca(usedSignals');
%pca_signals = (score*coeffs')';

%pca_signals = (coeffs'* (usedSignals));
pca_signals = ((usedSignals') * coeffs)';

% %perform ICA:
% [Zica, ~, ~, ~] = myICA(usedSignals, size(usedSignals, 1));
% pca_signals = Zica;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%CHANGE PCA BACK!!! ^

csvwrite('..\output\PCA_Signals.csv', pca_signals);

