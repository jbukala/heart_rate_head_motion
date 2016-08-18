%calculate HR by counting peaks in signal.
%signal is selected by detecting peaks and selecting signal with smallest
%stdev of inter-peak distances. Now break selected signal into 3 chunks,
%analyze each separately and use solutions that are closest together.
%return the mean of these two solutions.

%do this ONLY at thresh = 0.6:
thresh = 0.6;
chunks = 3; %number of pieces to cut the HR signal in

minPk = round((60*Fs)/150); %minimum peak distance corresponds to a 150bpm HR
num = csvread('..\output\PCA_Signals.csv');
%num = csvread('..\output\SSD_Signals.csv');
highestSig = 15; %max principal component to look at

%num = csvread('C:\Users\Joris\Desktop\PCA_Signals_test.csv');% REMOVE

for i=1:min(highestSig, size(num,1))
   [pks, ~]=findpeaks(num(i,:),'MinPeakDistance',minPk);
   threshold = mean(pks) * thresh;
   [pks,loc]=findpeaks(num(i,:),'MinPeakDistance',minPk, 'MinPeakHeight', threshold);
   v(i) = inf;
   freq(i) = inf;
   if length(loc)>1 %so doesn't divide by zero
       f1=mean(diff(loc)); 
       freq(i)=(60*Fs)/f1;
       v(i)=std(diff(loc))+rand*0.001;%just in case two stds are the same
   end
end

minIdx = find(min(v') == v'); %best HR-signal
bpm=freq(minIdx);
    
chunk_length = floor(size(num,2)/chunks);
for j=1:chunks %for each chunk
    start_chunk = 1+ (j-1) * chunk_length;
    stop_chunk = j * chunk_length;
    signal = num(minIdx, start_chunk:stop_chunk);
    
   [pks, ~]=findpeaks(signal,'MinPeakDistance',minPk);
   threshold = mean(pks) * thresh;
   [pks,loc]=findpeaks(signal,'MinPeakDistance',minPk, 'MinPeakHeight', threshold);
   p_freq(j) = inf;
   if length(loc)>1 %so doesn't divide by zero
       p_freq(j)= 60 * length(pks)*(Fs/length(signal));
   end
end

%get matrix of differences of results for each chunk. Use the two chunks
%with closest HR-result.
%then mean their HR-result and return that as the end-result
differences = inf * ones(chunks, chunks);
for i=1:chunks
   for j=i+1:chunks 
       differences(i,j) = abs(p_freq(i) - p_freq(j));
   end
end

[B,I]=min(differences(:)); %get smallest matrix element
[I,J] = ind2sub(size(differences),I); %and its 2D-index

bpm_chunk = mean([p_freq(I) p_freq(J)]);

threshresult = [bpm bpm_chunk];