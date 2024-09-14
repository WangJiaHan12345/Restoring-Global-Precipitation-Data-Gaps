%��Ԥ����ת��Ϊ������ʽ
%����ÿ���Ӧ������
FolderPath=input('���������ݴ洢�ļ���:','s'); 
index=strfind(FolderPath,'\');  %����ַ�'\'��FolderPath��λ��
Files=dir(FolderPath);
FilesCount=length(Files);


SaveFolder1=strcat('H:\ʱ��Ԥ��\�ĸ������������\shirun\result\������ʽ\','��\3-5\cpc'); %����ļ���·��
if exist(SaveFolder1,'dir')~=7  %���·�����������½�·��
    mkdir(SaveFolder1);
end

SaveFolder2=strcat('H:\ʱ��Ԥ��\�ĸ������������\shirun\result\������ʽ\','��\3-5\early'); %����ļ���·��
if exist(SaveFolder2,'dir')~=7  %���·�����������½�·��
    mkdir(SaveFolder2);
end

SaveFolder3=strcat('H:\ʱ��Ԥ��\�ĸ������������\shirun\result\������ʽ\','��\3-5\final'); %����ļ���·��
if exist(SaveFolder3,'dir')~=7  %���·�����������½�·��
    mkdir(SaveFolder3);
end

SaveFolder4=strcat('H:\ʱ��Ԥ��\�ĸ������������\shirun\result\������ʽ\','��\3-5\predict'); %����ļ���·��
if exist(SaveFolder4,'dir')~=7  %���·�����������½�·��
    mkdir(SaveFolder4);
end

% ��������Ԥ��Ľ��
count = 92*(FilesCount-2);
fid1 = fopen('H:\ʱ��Ԥ��\�ĸ������������\shirun\result\����\��\3-5\cpc\A_data.txt','rb','l');
data1 = cell2mat(textscan(fid1,'%f','headerlines',0));
data1 = reshape(data1,1,count);
data1 = data1'; 
fclose(fid1); 

fid2 = fopen('H:\ʱ��Ԥ��\�ĸ������������\shirun\result\����\��\3-5\early\A_data.txt','rb','l');
data2 = cell2mat(textscan(fid2,'%f','headerlines',0));
data2 = reshape(data2,1,count);
data2 = data2';
fclose(fid2); 

fid3 = fopen('H:\ʱ��Ԥ��\�ĸ������������\shirun\result\����\��\3-5\final\A_data.txt','rb','l');
data3 = cell2mat(textscan(fid3,'%f','headerlines',0));
data3 = reshape(data3,1,count);
data3 = data3';
fclose(fid3); 

fid4 = fopen('H:\ʱ��Ԥ��\�ĸ������������\shirun\result\����\��\3-5\predict\A_data.txt','rb','l');
data4 = cell2mat(textscan(fid4,'%f','headerlines',0));
data4 = reshape(data4,1,count);
data4 = data4';
fclose(fid4); 

disp('������...');


for k=3:FilesCount
     FilePath=strcat(FolderPath,'\',Files(k).name);  %�ļ�·��\�ļ���
     Name=Files(k).name;
     location=strfind(Name,'.');
      
      
     rain1 = data1(92*(k-3)+1:92*(k-2),1);
     rain2 = data2(92*(k-3)+1:92*(k-2),1);
     rain3 = data3(92*(k-3)+1:92*(k-2),1);
     rain4 = data4(92*(k-3)+1:92*(k-2),1);
     
     SaveFiles=strcat(Name(1:6),'.txt'); %CPC����ļ���·��
     
     outfile1=strcat(SaveFolder1,'\',SaveFiles);
     outfile2=strcat(SaveFolder2,'\',SaveFiles);
     outfile3=strcat(SaveFolder3,'\',SaveFiles);
     outfile4=strcat(SaveFolder4,'\',SaveFiles);

     
     fid1=fopen(outfile1,'w');
     for i=1:1:92  %����92 92 90 87  �ϣ�90 87 92 92 
         for j=1:1:1
             if j==1
                 fprintf(fid1,'%g\r\n',rain1(i,j));
             else
                 fprintf(fid1,'%g ',rain1(i,j));
             end
         end   
     end
     fclose(fid1);       
     
     fid2=fopen(outfile2,'w');
     for i=1:1:92  %����92 92 90 87  �ϣ�90 87 92 92 
         for j=1:1:1
             if j==1
                 fprintf(fid2,'%g\r\n',rain2(i,j));
             else
                 fprintf(fid2,'%g ',rain2(i,j));
             end
         end   
     end
     fclose(fid2);  
     
     fid3=fopen(outfile3,'w');
     for i=1:1:92  %����92 92 90 87  �ϣ�90 87 92 92
         for j=1:1:1
             if j==1
                 fprintf(fid3,'%g\r\n',rain3(i,j));
             else
                 fprintf(fid3,'%g ',rain3(i,j));
             end
         end   
     end
     fclose(fid3);  
     
     
     fid4=fopen(outfile4,'w');
     for i=1:1:92  % 
         for j=1:1:1
             if j==1
                 fprintf(fid4,'%g\r\n',rain4(i,j));
             else
                 fprintf(fid4,'%g ',rain4(i,j));
             end
         end   
     end
     fclose(fid4);  
end

disp('�������');