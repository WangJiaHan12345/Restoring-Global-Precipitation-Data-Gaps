%features
FolderPath=input('���������ݴ洢�ļ���:','s'); %�����������ı�׼
FolderPath1=input('���������ݴ洢�ļ���:','s'); %����������������� 
SaveFolder=strcat('H:\ʱ��Ԥ��\���\shirun\02_final_data\����\��\6-8\','ceshi_feature\wendu'); %����ļ���·��
if exist(SaveFolder,'dir')~=7  %���·�����������½�·��
    mkdir(SaveFolder);
end
Files=dir(FolderPath);
FilesCount=length(Files);

Files1=dir(FolderPath1);
FilesCount1=length(Files1);
disp('������...');

parfor k=3:FilesCount
      FilePath=strcat(FolderPath,'\',Files(k).name);  %�ļ�·��\�ļ���
      Name=Files(k).name;
      location=strfind(Name,'.');  %����ַ�'.'��FilePath��λ��
  
      i= str2num(Name(location(end)-6:location(end)-4));
      j= str2num(Name(location(end)-3:location(end)-1));
      
      data1=zeros(FilesCount1-2,1); %%%%12-2=272  3-5=276  6-8=276 9-11=273
      
      for m=3:FilesCount1  
          FilePath1=strcat(FolderPath1,'\',Files1(m).name);  %�ļ�·��\�ļ���
          fid=fopen(FilePath1,'rb','l');  % 'rb'�Զ����Ʒ�ʽֻ�����ʹ��ļ���Ҳ����ֱ��'r';'l':little endianС�����
          data = cell2mat(textscan(fid,'%f','headerlines',6));
          data = reshape(data,720,240);
          data = data'; 

          data1(m-2,1) = data(i,j);
          fclose(fid); 
      end

     SaveFiles=strcat(Name(1:location(end)-1),'.txt');

     outfile=strcat(SaveFolder,'\',SaveFiles);

     if exist(outfile,'file')~=0 
        delete(outfile);     
     end
     fid1=fopen(outfile,'w');


     for i=1:1:FilesCount1-2
         for j=1:1:1
             if j==1
                 fprintf(fid1,'%g\r\n',data1(i,j));
             else
                fprintf(fid1,'%g ',data1(i,j));
             end
         end   
     end
     fclose(fid1); 
    
end
disp('�������');
