% fclose('all');
% Path = 'J:\【04】站点数据\中国自动站与CMORPH降水产品融合的逐时降水量网格数据集(1.0版)\2017\2017\'; 
% filena1='SEVP_CLI_CHN_MERGE_CMP_PRE_HOUR_GRID_0.10-';
% filena2=2017010100;
% filena3=2017010100;
% odir='F:\newput\cmpa\2017\';
% odir1='F:\output\true\';
% odir2='F:\output\staion\';
% B=load('F:\input\mask01.txt');
% E=load('F:\input\staion.txt');
% C=zeros(150,400);
% A=C;
% AA=A;
% DD=A;
% D=A;
% F=[];header=[];
% num=0;
% for m=1:24*365
%     filena4=fix(filena2/10000);
% 
%     if mod(filena2,100)<23
%     filena2=filena2+1;
%     else 
%     filena2=filena2+100-23;
%     end
% if mod(fix(filena2/100),100)>daytime(filena4)
%     filena3=filena3+10000;
%     filena2=filena3;
% end
%     f=mat2str(filena2);
%    filena=strcat(Path,filena1,f,'.grd');
%    if ~exist (filena,'file')
%        A=zeros(150,400);
%    else
%       fid=fopen(filena,'r');
%       rain=fread(fid,[700,880],'float'); 
%       rain=rain';
%         preci=rain(1:440,:);
%         cha=zeros(440,50);
%         preci=[cha preci(:,1:350)];
%         preci=preci(101:250,:);
%         preci(preci<0)=0;
%         preci=flipud(preci);
%         for i=1:150
%         for j=1:400
%         if E(i,j)==0
%         A(i,j)=-1;
%         else
%         A(i,j)=preci(i,j);
%         end
%         end
%         end
%    end
%         E=E+A;
%         if mod(m,24)==0
%         E(E==0)=100;
%         E(E<0)=0;
%         [h,l,D]=find(E);
%         D(D==100)=0;
% %         n=mat2str(m/24);
%         miao=strcat(odir,'cmpa','.txt');
%         fid=fopen(miao,'a');
%         fprintf(fid,'%.2f\t',D);
%         E=zeros(150,400);
%         end
% end

   
    

Path='J:\转移\2017\';
File = dir(fullfile(Path,'*.grd')); % 显示文件夹下所有符合后缀名为.txt文件的完整信息
FileNames = {File.name}'; 
Length_Names = size(FileNames,1); 
odir='F:\newput\cmpa\2017\';
odir1='F:\output\true\';
B=load('F:\input\mask01.txt');
% E=load('F:\input\staion.txt');
C=zeros(150,400);
A=C;
AA=A;
DD=A;
D=A;
F=[];
num=0;
aa=zeros(1000,1);
sum=zeros(150,400);
header=[];
for k = 1 : Length_Names
input=strcat(Path, FileNames(k));
% name1=FileNames(k);
% name=name1{1,1};
fid=fopen(input{1,1},'r');
rain=fread(fid,[700,880],'float');
rain=rain';
preci=rain(1:440,:);
% staion=rain(441:880,:);
cha=zeros(440,50);
preci=[cha preci(:,1:350)];
preci=preci(101:250,:);
preci(preci<0)=0;
preci=flipud(preci);
% staion=[cha staion(:,1:350)];
% staion=staion(101:250,:);

% staion=flipud(staion);
for i=1:150
    for j=1:400
        if B(i,j)<0%5||E(i,j)==0
        A(i,j)=-1;
        D(i,j)=0;
        else
        A(i,j)=preci(i,j);
%            D(i,j)=staion(i,j);
%         A(i,j)= preci((2*i-1),(2*j-1))+preci((2*i-1),(2*j))+preci((2*i),(2*j-1))+preci((2*i),(2*j));
%         A(i,j)= A(i,j)/4;
%         D(i,j)= staion((2*i-1),(2*j-1))+staion((2*i-1),(2*j))+staion((2*i),(2*j-1))+staion((2*i),(2*j));
%         n=staion((2*i-1),(2*j-1))+staion((2*i-1),(2*j))+staion((2*i),(2*j-1))+staion((2*i),(2*j));
%         if n>0
%         A(i,j)=A(i,j)/n;
%         end
        end
    end
end
sum=sum+A;
if mod(k,24)==0
   num=num+1;
   numm=num2str(num);
   OUTPUT(odir,numm,header,sum); 
   sum=zeros(150,400);
end
% aa(k,1)=sum(D(:));
% AA=AA+A;
% if mod(k,24)==0
% % AA(AA==0)=100;
% AA(AA<0)=-1;
% num=num+1;
% numm=num2str(num);
% % OUTPUT(odir,numm,header,AA);
% AA=zeros(150,400);
fclose('all');
end
% end

% AA(AA<0)=-9999;
% AA(:,1:50)=[];
% header=['ncols           350';
%         'nrows           150';
%         'xllcorner        70';
%         'yllcorner        25';
%         'cellsize        0.1';
%         'NODATA_value  -9999']; 
% OUTPUT(odir,'cmpa',header,AA);
% OUTPUT(odir,'staion1',header,D);