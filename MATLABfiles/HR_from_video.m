function [heartRate] = HR_from_video(filename, sessionNo, Fs)
%Given a folder, video-id and video framerate as input, return a vector of
%heart rate results using several algorithms

csvname= '';
csvPresent = 0;

if ~strcmp(filename(end-3:end), '.avi')%if not filename but folder is given, find an avi file
    currFolder = pwd;
    cd(filename);
    avifiles = dir('*.avi');
    mp4files = dir('*.mp4');
    
    csvfiles = dir('*.csv');
    sessionFolder = filename;
    
    if length(avifiles) < 1 && length(mp4files) < 1 && length(csvfiles)  < 1 %if theres no avi-file or csv-file
        heartRate = 1;
        cd(currFolder);
        disp('Cant find an avi/mp4-file or a csv-file');
        return;
    end
    
    if length(csvfiles) >= 1
        csvname = strcat(sessionFolder, 'rawSignals.csv'); %change this line to select only LM or only GFT points. rawSignals is BOTH 
        csvPresent = 1;
    end
    
    if length(avifiles) >= 1
        filename = strcat(sessionFolder, avifiles(1).name);
    elseif length(mp4files) >= 1
        filename = strcat(sessionFolder, mp4files(1).name);
    else
        filename = 'no_video_file';
    end;
    cd(currFolder);
end

%Perform tracking & filtering in C++:

%skip tracking if the raw signals are already saved
if ~csvPresent
    sys_command = strcat('..\pulsefromheadmotion-master\bin\Release\pfhmain.exe "', filename, ['" ' num2str(sessionNo)]);
    ret = system(sys_command);
    if (ret)
        disp('Something went wrong with executing the C++ tracking and/or analysis')
        dlmwrite('..\output\cpp_results.csv', [sessionNo 1 1], '-append');
    end
    
    %copy tracking result csv to the folder containing the video
    copyfile('..\output\rawSignals.csv', strcat(sessionFolder, 'rawSignals.csv'));
    copyfile('..\output\GFT_signals.csv', strcat(sessionFolder, 'GFT_signals.csv'));
    copyfile('..\output\landmark_signals.csv', strcat(sessionFolder, 'landmark_signals.csv'));
    
    if Fs == 61
    %copy tracking results to separate MAHNOB-folder
    mkdir('..\..\MAHNOB', num2str(sessionNo));
    copyfile('..\output\rawSignals.csv', strcat('..\..\MAHNOB\', num2str(sessionNo), '\', 'rawSignals.csv'));
    copyfile('..\output\GFT_signals.csv', strcat('..\..\MAHNOB\', num2str(sessionNo), '\', 'GFT_signals.csv'));
    copyfile('..\output\landmark_signals.csv', strcat('..\..\MAHNOB\', num2str(sessionNo), '\', 'landmark_signals.csv'));
    elseif Fs == 30
    %copy tracking results to separate selfmadeDB-folder
    mkdir('..\..\selfmadeDB', num2str(sessionNo));
    copyfile('..\output\rawSignals.csv', strcat('..\..\selfmadeDB\', num2str(sessionNo), '\', 'rawSignals.csv'));
    copyfile('..\output\GFT_signals.csv', strcat('..\..\selfmadeDB\', num2str(sessionNo), '\', 'GFT_signals.csv'));
    copyfile('..\output\landmark_signals.csv', strcat('..\..\selfmadeDB\', num2str(sessionNo), '\', 'landmark_signals.csv'));
    end
     
    csvname = strcat(sessionFolder, 'rawSignals.csv');
end

% Perform cutting signal into pieces and clustering them:
%'HR_from_filtered_sig()
% run('KMeans_test.m')
% run('SOM_test.m')


% HR_from_raw_sig(csvname, Fs); %use SOM + k-means for signal analysis. if
% using this, comment out the signal selection in C++ (the next 11 lines of code)

% only analyze (do PCA & signal selection in C++):
if csvPresent
    copyfile(strcat(sessionFolder, 'rawSignals.csv'), '..\output\rawSignals.csv');
    copyfile(strcat(sessionFolder, 'GFT_signals.csv'), '..\output\GFT_signals.csv');
    copyfile(strcat(sessionFolder, 'landmark_signals.csv'), '..\output\landmark_signals.csv');
    
    sys_command = strcat('..\pulsefromheadmotion-master\bin\Release\pfhmain.exe "', filename, ['" ' num2str(sessionNo) ' 1']);
    ret = system(sys_command);
    if (ret)
        disp('Something went wrong in the C++ signal analysis')
        dlmwrite('..\output\cpp_results.csv', [sessionNo 1 1], '-append');
    end
end


%run('peakdetect.m')
run('peakdetect_pieces.m')
run('DCT_heartrate.m')

heartRate = [threshresult DCT_HR];
end