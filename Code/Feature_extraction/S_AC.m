clear all
clc

load('N_proteinA.mat')
load('N_proteinB.mat')
load('P_proteinA.mat')
load('P_proteinB.mat')
load('SVHEHS.mat')
OriginData=SVHEHS;
OriginData=OriginData';
num1=numel(P_proteinA);

for lag=1:11
   S_AC_Pa=[];
   S_AC_Pb=[];
   S_AC_Na=[];
   S_AC_Nb=[];
   for i=1:num1
        [AC_Pa,AC_Pb]=AC(P_proteinA{i},P_proteinB{i},OriginData,lag);
        [AC_Na,AC_Nb]=AC(proteinA{i},proteinB{i},OriginData,lag);
        S_AC_Pa=[S_AC_Pa;AC_Pa];
        S_AC_Pb=[S_AC_Pb;AC_Pb];
        S_AC_Na=[S_AC_Na;AC_Na];
        S_AC_Nb=[S_AC_Nb;AC_Nb];
   end
   
   data_AC=[[S_AC_Pa,S_AC_Pb];[S_AC_Na,S_AC_Nb]];
   data_AC=[[ones(5594,1);zeros(5594,1)],data_AC];
  
   total = ['T_S_AC_' num2str(lag) '.mat'];
   save(total,'data_AC')  

   divide = ['D_S_AC_' num2str(lag) '.mat'];
   save(divide,'S_AC_Pa','S_AC_Pb','S_AC_Na','S_AC_Nb') 
end
