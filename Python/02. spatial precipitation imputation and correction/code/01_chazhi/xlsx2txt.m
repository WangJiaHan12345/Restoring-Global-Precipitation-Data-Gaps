% This program is to read excel to txt 
% The author is Zhanyang Zhang
tic
fi = xlsread('D:\03Paper\2015Jinshajiang\data\2015金沙江下段水系_1.xlsx');

fid = fopen('D:\03Paper\2015Jinshajiang\data\2015金沙江下段水系_1.txt','w');
fprintf(fid,strcat(repmat('%9.4f\t',1,99),'\r\n'),fi');
fclose(fid);

toc