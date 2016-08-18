%clc
%clear all
for j=1:1 %RANGE OF VIDEO FILES (STARTING FROM 2?)
    %num = xlsread( 'C:\Users\i6093682\Desktop\output\signalSelectionResults.xlsx' , j+1);
    %num = xlsread( 'C:\Users\i6093682\Desktop\output\signalSelectionEmotions.xlsx' , j);
    %num = csvread('C:\Users\i6093682\Desktop\output\trainingSignals.csv');
    %num = xlsread( 'C:\Users\i6093682\Desktop\output\signalSelectionEmotions.xlsx' , j);
    
    Fs = 61; %sampling rate
    %Cut signals into pieces:
    T = 10; %timelength in seconds of piece
    data = csvread('..\output\filteredSignals.csv');
    %data = xlsread( 'C:\Users\i6093682\Desktop\output\filterSignalEmotions.xlsx' , 4);
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
    colors = rand(k,3) * 0.8;
    dim1 = 1; %dimensions to plot
    dim2 = 2;
    
    quality = zeros(k,1);
%     figure(1),
%     subplot(2,1,1)
    for i=1:k
%         points = weights(idx==i,:);
%         plot(points(:,dim1), points(:,dim2), '*', 'color',colors(i,:))
%         hold on;
%         plot(C(i,dim1), C(i,dim2), 'o', 'color',colors(i,:)) %plot cluster centre
%         hold on;
        %quality(i) = norm(C(i,:)); %Selection criteria to minimize to choose which cluster contains the clean signals
        
        minClasses = (idx==i); %SOM-classes which are in the selected k-means cluster
        used = (minClasses(classes)==1); %signal-pieces
%         quality(i) = sum(used); %Selection criteria to maximize to choose which cluster contains the clean signals
        quality(i) = sum(used); %Selection criteria to maximize to choose which cluster contains the clean signals
    end
    
    [~ , minIdx] = max(quality); %now minIdx is the cluster with a center of smallest norm
    points = weights(idx==minIdx,:); %select 'best' cluster
%     subplot(2,1,2)
%     plot(points(:,dim1), points(:,dim2), '*', 'color', colors(minIdx,:))
%     hold on;
%     plot(C(minIdx,dim1), C(minIdx,dim2), 'ko') %plot cluster centre
%     axis equal
end

minClasses = (idx==minIdx); %SOM-classes which are in the selected k-means cluster
used = (minClasses(classes)==1); %signal-pieces to use
%unused = (minClasses(classes)==0);
usedSignals = num(used,:);
%unusedSignals = num(unused,:);

% figure(2),
% clf;
% subplot(2,1,1)
% plot(usedSignals(95:99,:)', 'b'); 
% hold on; 
% subplot(2,1,2)
% plot(unusedSignals(95:99,:)', 'r')
% title('Red signals seem noisy')

csvwrite('..\output\SOM_signals.csv', usedSignals)
