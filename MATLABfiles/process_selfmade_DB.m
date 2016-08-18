%Determines Heart-rate for a whole set of videos and saves these values in one matrix to
%easily compare effectiveness of several methods 

%Needs the folder which
%contains one sub-folder for each video. In this sub-folder there should
%either be the video-file in mp4 or avi-format, or a csv file containing
%time-series of the tracked points. The sub-folder should be given a unique number
%as a name.
format shortG

%path = 'N:\FHS_DKE\Heart rate from head motions\webcamrecordings\'; %self-made database on network-drive
path = '..\..\selfmadeDB\'; %self-made DB time-series on desktop pc

delete('..\output\CPP_results.csv');

Fs = 30; %sampling freq of all self-made videos
%Fs = 250; %to follow Balakrishnan

currFolder = pwd;
cd(path);
listing = dir();
cd(currFolder);

%Make a list of all the sub-folders:
for i=3:size(listing,1) %starts at 3 because of hidden windows folders stuff that have to be ignored.. (thumbs.db as well at the end if the folder is opened in windows..)
    session(i-2) = str2double(listing(i).name);
end
session = sort(session);

GT_HR = csvread('..\selfmade_DB_ground_truth.csv'); %retrieve ground truth from csv-file on disk

%for each sub-folder:
for i=1:size(session,2)
    foldername = strcat(path, num2str(session(i)), '\');
    HeartRate(i) = GT_HR(i,2);
    
    video_HeartRate(i, :) = HR_from_video(foldername, session(i), Fs);
    
    disp([session(i) HeartRate(i) video_HeartRate(i, :)]);
end

cpp_results = csvread('..\output\CPP_results.csv');
delete('..\output\CPP_results.csv');

result = [[session; HeartRate; video_HeartRate']' cpp_results(:,2) cpp_results(:,3)];

predictions = result(:,6); %chosen method of calculation to use for further plots/results
errors = (abs(result(:,2) - predictions)./result(:,2))*100;

%Calc Pearson corr. coeff.
[RHO, PVAL] = corr(result(:,2), predictions, 'tail', 'right');

accuracy = sum(errors < 10)/size(result,1);
mean_abs_err = mean(errors);
plot(result(:,2), predictions, 'r*')
title(['Self-made   Mean abs err: ' num2str(mean_abs_err) '% correlation: ' num2str(RHO) ' p-value: ' num2str(PVAL)])
xlabel('Ground truth HR (bpm)')
ylabel('Predicted HR (bpm)')

%disp('      Session       ECG_HR       Thresh1      Thresh2      Thresh3    Thresh4      DCT      FFT      DCT2'); %using peakdetect.m
disp('      Session       ECG_HR       Thresh1      Thresh_chunk   DCT         FFT       DCT2'); %for peakdetect_pieces.m
disp(result);