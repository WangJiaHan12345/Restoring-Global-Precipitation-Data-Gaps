
day =92; %3-5 6-8:368  9-11:364  12-2:361
grids =11284;

fid=fopen('G:\ȫ��\ʱ��Ԥ����\global_ns\ȫ������\global_s\3-5\ANN_sum_rain.txt','rb');  % 'rb'�Զ����Ʒ�ʽֻ�����ʹ��ļ���Ҳ����ֱ��'r';'l':little endianС�����
x = cell2mat(textscan(fid,'%f','headerlines',0));
x = reshape(x,grids,1);
x = x/day;   
fclose(fid);

fid1=fopen('G:\ȫ��\ʱ��Ԥ����\global_ns\ȫ������\global_s\3-5\cpc_sum_rain.txt','rb');  % 'rb'�Զ����Ʒ�ʽֻ�����ʹ��ļ���Ҳ����ֱ��'r';'l':little endianС�����
x1 = cell2mat(textscan(fid1,'%f','headerlines',0));
x1 = reshape(x1,grids,1);
x1 = x1/day;   
fclose(fid1);

fid2=fopen('G:\ȫ��\ʱ��Ԥ����\global_ns\ȫ������\global_s\3-5\Early_sum_rain.txt','rb');  % 'rb'�Զ����Ʒ�ʽֻ�����ʹ��ļ���Ҳ����ֱ��'r';'l':little endianС�����
x2 = cell2mat(textscan(fid2,'%f','headerlines',0));
x2 = reshape(x2,grids,1);
x2 = x2/day;   
fclose(fid2);

fid3=fopen('G:\ȫ��\ʱ��Ԥ����\global_ns\ȫ������\global_s\3-5\Final_sum_rain.txt','rb');  % 'rb'�Զ����Ʒ�ʽֻ�����ʹ��ļ���Ҳ����ֱ��'r';'l':little endianС�����
x3 = cell2mat(textscan(fid3,'%f','headerlines',0));
x3 = reshape(x3,grids,1);
x3 = x3/day;
fclose(fid3);

SaveFolder=strcat('G:\ȫ��\ʱ��Ԥ����\global_ns\ȫ������\��׼����\','��\3-5'); %����ļ���·��
if exist(SaveFolder,'dir')~=7  %���·�����������½�·��
    mkdir(SaveFolder);
end


outfile1=strcat(SaveFolder,'\','ANN_sum_rain.txt');
outfile2=strcat(SaveFolder,'\','cpc_sum_rain.txt');
outfile3=strcat(SaveFolder,'\','Early_sum_rain.txt');
outfile4=strcat(SaveFolder,'\','Final_sum_rain.txt');


fid1=fopen(outfile1,'w');
fid2=fopen(outfile2,'w');
fid3=fopen(outfile3,'w');
fid4=fopen(outfile4,'w');
        

 for i=1:1:grids
     for j=1:1:1
         if j==1
             fprintf(fid1,'%g\r\n',x(i,j));
         else
            fprintf(fid1,'%g ',x(i,j));
         end
     end   
 end
         
 for i=1:1:grids
     for j=1:1:1
         if j==1
             fprintf(fid2,'%g\r\n',x1(i,j));
         else
            fprintf(fid2,'%g ',x1(i,j));
         end
     end   
 end         
         
 for i=1:1:grids
     for j=1:1:1
         if j==1
             fprintf(fid3,'%g\r\n',x2(i,j));
         else
            fprintf(fid3,'%g ',x2(i,j));
         end
     end   
 end         
         
 for i=1:1:grids
     for j=1:1:1
         if j==1
             fprintf(fid4,'%g\r\n',x3(i,j));
         else
            fprintf(fid4,'%g ',x3(i,j));
         end
     end   
 end            
         
fclose(fid1); 
fclose(fid2); 
fclose(fid3); 
fclose(fid4); 