%calculate HR by counting peaks in signal.
%signal is selected by detecting peaks and selecting signal with smallest
%standard deviation of inter-peak distances

thresh = [0.4 0.5 0.55 0.6];

%MAHNOB
%Fs = 61;
%self-made:
%Fs = 30;

minPk = round((60*Fs)/150); %minimum peak distance corresponds to a 150bpm HR
num = csvread('..\output\PCA_Signals.csv');

highestSig = 15; %max principal component to look at

for j=1:size(thresh,2) %for each threshold
    for i=1:min(highestSig, size(num,1))
        [pks, ~]=findpeaks(num(i,:),'MinPeakDistance',minPk);
        threshold = mean(pks) * thresh(j);
        [pks,loc]=findpeaks(num(i,:),'MinPeakDistance',minPk, 'MinPeakHeight', threshold);
        v(i) = inf;
        freq(i) = inf;
        if length(loc)>1 %so doesn't divide by zero
            f1=mean(diff(loc)); %using MEAN
            %f1=median(diff(loc)); %using MEDIAN
            freq(i)=(60*Fs)/f1;
            v(i)=std(diff(loc))+rand*0.001;%just in case two stds are the same
        end
    end

    minIdx = find(min(v') == v');
    bpm(j)=freq(minIdx);
end

threshresult = bpm;