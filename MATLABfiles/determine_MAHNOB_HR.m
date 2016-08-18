%Determines Heart-rate for a whole set of videos and saves these values in one matrix to
%easily compare effectiveness of several methods 

%Needs the folder which
%contains one sub-folder for each video. In this sub-folder there should
%either be the video-file in mp4 or avi-format, or a csv file containing
%time-series of the tracked points. The sub-folder should be given a unique number
%as a name.

format shortG

path = '..\..\MAHNOB\'; %contains the time-series for the MAHNOB-HCI dataset
%path = 'N:\FHS_DKE\Heart rate from head motions\Sessions\'; %whole relevant MAHNOB-DB videos and time-series on network drive

delete('..\output\CPP_results.csv');

Fs = 61; %sampling freq of all mahnob-hci videos
%Fs = 250; %To replicate Balakrishnan-paper use this, they upsample the
%signals to 250Hz

currFolder = pwd;
cd(path);
listing = dir();
cd(currFolder);

%Make a list of all the sub-folders:
for i=3:size(listing,1) %starts at 3 because of hidden windows folders stuff that have to be ignored..
    session(i-2) = str2double(listing(i).name);
end
session = sort(session);

%for each sub-folder:
for i=1:size(session,2)
    foldername = strcat(path, num2str(session(i)), '\');
    HeartRate(i) = HR_from_BDF_file(foldername, Fs); %Retrieve HR from an ECG contained in a .bdf-file. Uses the Pan-Tompkins algorithm for this
    
    if HeartRate(i) < 50
        disp(strcat('HR in session ', num2str(session(i)), ' is lower than 50. Is something wrong with the ECG?'))
    end
    if HeartRate(i) > 200
        disp(strcat('HR in session ', num2str(session(i)), ' is higher than 200. Is something wrong with the ECG?'))
    end
    
    video_HeartRate(i, :) = HR_from_video(foldername, session(i), Fs); %Uses several ways to infer heart-rate from video images
    
    disp([session(i) HeartRate(i) video_HeartRate(i, :)]);
end

cpp_results = csvread('..\output\CPP_results.csv'); %read in all results for every video calculated in the c++-program
delete('..\output\CPP_results.csv');
csvwrite('..\output\PanTompkins_HR.csv', [session; HeartRate]'); %save HR from this session in a separate file

GT_HR = csvread('..\output\GT_HR.csv')'; %read corrected ground-truth HR thats already saved (manually corrected for subject 23 and other errors, Pan-Tompkins did poorly on some signals)
session = GT_HR(1,:);
HeartRate = GT_HR(2,:);

result = [[session; HeartRate; video_HeartRate']' cpp_results(:,2) cpp_results(:,3)];
%result = [session; HeartRate; video_HeartRate']';

predictions = result(:,3); %chosen method of calculation to use for further plots/results
errors = (abs(result(:,2) - predictions)./result(:,2))*100;

%Calc Pearson corr. coeff.
[RHO, PVAL] = corr(result(:,2), predictions, 'tail', 'right');

accuracy = sum(errors < 10)/size(result,1);
mean_abs_err = mean(errors);
plot(result(:,2), predictions, 'r*')
title(['MAHNOB  Mean abs err: ' num2str(mean_abs_err) '% correlation: ' num2str(RHO) ' p-value: ' num2str(PVAL)])
xlabel('Ground truth HR (bpm)')
ylabel('Predicted HR (bpm)')

%disp('      Session       ECG_HR       Thresh1      Thresh2      Thresh3      Thresh4      DCT     cpp_FFT    cpp_DCT2');
disp('      Session       ECG_HR       Thresh      Thesh_pieces   DCT      cpp_FFT    cpp_DCT2');

disp(result);