% ��������ʽתΪ��������ʽ
FolderPath1=input('���������ݴ洢�ļ���:','s');   %����GPCC�ļ���
index1=strfind(FolderPath1,'\');  %����ַ�'\'��FolderPath��λ��
Files1=dir(FolderPath1);
FilesCount1=length(Files1);

SaveFolder=strcat('G:\ȫ��\�ռ�У�����\���ں�GPCP�Ƚ�\','�³߶�\DNN'); %����ļ���·��
if exist(SaveFolder,'dir')~=7  %���·�����������½�·��
    mkdir(SaveFolder);
end

disp('������...');


parfor k=3:1:FilesCount1
     
      FilePath1=strcat(FolderPath1,'\',Files1(k).name);  %�ļ�·��\�ļ���
      Name=Files1(k).name;
      location=strfind(Name,'.');  %����ַ�'.'��FilePath��λ��


      fid1=fopen(FilePath1,'rb','l');  % 'rb'�Զ����Ʒ�ʽֻ�����ʹ��ļ���Ҳ����ֱ��'r';'l':little endianС�����
      data1 = cell2mat(textscan(fid1,'%f','headerlines',0));
      data1 = reshape(data1,1,1461);
      data1 = data1'; 
      fclose(fid1);
      
     result = zeros(48,1);
    
     day = [0,31,59,90,120,151,181,212,243,273,304,334,365,396,425,456,486,517,547,578,609,639,670,700,731,762,790,821,851,882,912,943,974,1004,1035,1065,1096,1127,1155,1186,1216,1247,1277,1308,1339,1369,1400,1430,1461];
     
     for m = 1:1:48
         sum = 0;
         for n = day(m)+1:1:day(m+1)
             sum = sum + data1(n,1);
         end
         result(m,1)=sum;
     end
     
     SaveFiles=strcat(Name(1:6),'.txt'); %CPC����ļ���·��

     outfile=strcat(SaveFolder,'\',SaveFiles);

     if exist(outfile,'file')~=0 
        delete(outfile);     
     end    

    fid1=fopen(outfile,'w');

     for a=1:1:48
         for b=1:1:1
             if b==1
                 fprintf(fid1,'%g\r\n',result(a,b));
             else
                 fprintf(fid1,'%g ',result(a,b));
             end
         end   
     end
    fclose(fid1); 
end
