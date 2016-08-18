function [HR] = HR_from_BDF_file(filename, Fs)
%Gets the heartrate by reading in a BDF-file using the BioSig toolbox
%(http://biosig.sourceforge.net/download.html).
%Then uses the ECG signals in the BDF-file as input for the Pan-Tompkins
%algorithm to determine average heart-rate in a signal of 90 secs (or shorter if the signal itself is shorter)

%Relies on that the ECG-signals always have the same positions within the
%BDF files and the sampling frequency being 256 (used on MAHNOB-HCI
%database files).
%Gets the total ECG signal by summing all the leads

if ~strcmp(filename(end-3:end), '.bdf')%if not filename but folder is given, find a bdf file
    currFolder = pwd;
    cd(filename);
    bdffiles = dir('*.bdf');
    
    if length(bdffiles) < 1 %if theres no BDF-file
        HR = 1;
        disp('Cant find a DBF-file to read');
        cd(currFolder);
        return
    end
    filename = strcat(filename, bdffiles(1).name);
    cd(currFolder);
end

fs = 256;
T = 60; %signal length in seconds

[hdr, H1, h2] = sopen(filename);
[S, HDR] = sread(hdr, T);
sclose(HDR);
%ECG = sum(S(:,33:40),2)'; %use whole ECG-signal
%ECG = S(:,34)'; %use only ECG2-lead

startPos = floor(306 * Fs/61);
endPos = floor(2135 * Fs/61);
ECG = S(startPos:endPos,34)';%use the equivalent of frames 306 - 2135 from MAHNOB...
HR=1;

[ ~,qrs_i_raw, ~] = pan_tompkin(ECG',fs,0); %last argument is flag to plot or not, but crashes when plotting..
if (length(qrs_i_raw) > 1)
beat_time = (qrs_i_raw(2:end) - qrs_i_raw(1:end-1))/fs;
HR = 60/mean(beat_time);
end

if((HR < 40) || (HR > 130)) %if this is wrong discard first 20 secs, to make sure ECG2 started sampling!
    ECG2 = S(20*fs:end,34)';
    [ ~,qrs_i_raw, ~] = pan_tompkin(ECG2',fs,0); %last argument is flag to plot or not, but crashes when plotting..
    beat_time2 = (qrs_i_raw(2:end) - qrs_i_raw(1:end-1))/fs;
    HR2 = 60/mean(beat_time2);
    HR = HR2;
    disp('Discarding first 20 secs of ECG2');
end

end