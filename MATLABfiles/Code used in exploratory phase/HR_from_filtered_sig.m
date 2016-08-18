function HR_from_filtered_sig()

Fs = 61; %sampling rate
    %Cut signals into pieces:
    T = 10; %timelength in seconds of piece
    data = csvread('..\output\filteredSignals.csv');
    num = zeros(ceil(size(data,1)*size(data,2)/(T*Fs)), T*Fs);
    colptr = 1;
    rowptr = 1;
    piece = 1; %how many segments are there
    
    while(colptr < size(data,1))
        while(size(data,2) - rowptr >= (T*Fs))
            num(piece,:) = data(colptr,rowptr:(rowptr+(T*Fs)-1)) - data(colptr,rowptr); 
            rowptr = rowptr+T*Fs;
            piece = piece+1;
        end
        colptr = colptr+1;
        rowptr = 1;
    end
    num = num(1:piece-1,:); %grab the filled part of the matrix as data
    
    %Calculate signals entropy:
    entro = zeros(size(num,1), 1);
    for i=1:size(num,1)
       entro(i) = entropy(num(i,:)); 
    end
    
    %calculate signals max frequency content:
    
    maxFreq = zeros(size(num,1), 1);
    for i=1:size(num,1)
        sizeM = size(num);
        L = sizeM(2);
        Y = fft(num(i,:));
        P2 = abs(Y/L);
        P1 = P2(1:L/2+1);
        P1(2:end-1) = 2*P1(2:end-1);
        f = Fs*(0:(L/2))/L;
        
        [~, max_idx] = max(P1(2:end-1));
        maxFreq(i) = f(max_idx+1);
    end
    
    %Start Self-Organizing Map part:
    x = [abs(mean(num, 2)) std(num, 0, 2) entro maxFreq]';
    %x = [std(num, 0, 2) entro]';
    net = selforgmap([10 10]);
    net.trainParam.showWindow = false;
    net.trainParam.showCommandLine = false; 
    net = train(net,x);
    y = net(x);
    classes = vec2ind(y)';
    weights = net.IW{1,1};
    
    k=3;
    [idx,C] = kmeans(weights, k);
    
    quality = zeros(k,1);

    for i=1:k
        %minClasses = (idx==i); %SOM-classes which are in the selected k-means cluster
        %used = (minClasses(classes)==1); %signal-pieces
        %quality(i) = sum(used); %Selection criteria to MAXIMIZE to choose
        %which cluster contains the clean signals (select biggest cluster)
        quality(i) = -norm(C(i,:)); %Selection criteria to MAXIMIZE to choose which cluster contains the clean signals (smallest-norm cluster)
    end
    
    [~, minIdx] = max(quality); %now minIdx is the cluster with a center of smallest norm

minClasses = (idx==minIdx); %SOM-classes which are in the selected k-means cluster
used = (minClasses(classes)==1); %signal-pieces to use
%unused = (minClasses(classes)==0);
usedSignals = num(used,:);

csvwrite('..\output\SOM_signals.csv', usedSignals)
