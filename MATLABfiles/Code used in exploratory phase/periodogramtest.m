   Fs = 30;
   
   num = csvread('..\output\PCA_Signals.csv');
    
   x = num(1,:);
   [Pxx,Fxx] = periodogram(x,[],length(x),Fs);
   Y = max(Pxx);
   Fkappa = Y./mean(Pxx(2:end-1)) 
   
   plot(Fxx, Pxx)
    figure,
   
   x = num(2,:);
   [Pxx,Fxx] = periodogram(x,[],length(x),Fs);
   Y = max(Pxx);
   Fkappa = Y./mean(Pxx(2:end-1)) 
   plot(Fxx, Pxx, 'r')