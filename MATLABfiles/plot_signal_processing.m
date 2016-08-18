rawsig = csvread('..\..\MAHNOB\152\rawSignals.csv');
rawsig = rawsig(:,1:5000);
Fs = 61;
% t = 0:1/Fs:((length(rawsig)-1)/Fs);
% plot(t,rawsig(1,:))
% hold on
% plot(t,rawsig(10,:), 'r')
% hold on
% plot(t,rawsig(120,:), 'k')

HR = HR_from_video('C:\Users\Joris\Desktop\MasterThesis\MAHNOB\152\', 0, Fs);
pcasig = csvread('..\output\PCA_Signals.csv');
pcasig = pcasig(:,1:end);

t = 0:1/Fs:((length(pcasig)-1)/Fs);
plot(t,pcasig(1,:))
% hold on
% plot(t,pcasig(10,:), 'r')
% hold on
% plot(t,pcasig(12,:), 'k')