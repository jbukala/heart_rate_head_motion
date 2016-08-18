function [deltaf, deltaf2] = gaussfitSSD(ff,nmmt)


[pks,lcs] = findpeaks(nmmt);
[~,inpks] = sort(pks,'descend');
in1 = lcs(inpks(1));
in2 = lcs(inpks(2));

is1 = find(nmmt(in1+1:end)<2/3*nmmt(in1),1,'first');
is2 = find(nmmt(in2+1:end)<2/3*nmmt(in2),1,'first');

estsig1 = abs(ff(in1)-ff(in1+is1));

if isempty(is2)
    estsig2 = 4*abs(ff(in1)-ff(in2));
else
    estsig2 = abs(ff(in2)-ff(in2+is2));
end

ff_in1 = -((ff-ff(in1)).^2);
ff_in2 = -((ff-ff(in2)).^2);
ff_m_in1_in2 = -((ff-0.5*(ff(in1)+ff(in2))).^2);

Phi = @(x)(nmmt-(x(1)*exp(ff_in1/(2*(x(4))^2))+x(2)*exp(ff_in2/(2*(x(5))^2))+x(3)*exp(ff_m_in1_in2/(2*(x(6))^2))));
x0 = [nmmt(in1)/2 nmmt(in2)/2 nmmt(round(mean([in1,in2])))/4 estsig1 estsig2 4*abs(ff(in1)-ff(in2))];
[x,~,~]=LMFsolve(Phi,x0);

deltaf = abs(x(4))*2.5;
deltaf2 = abs(x(6));