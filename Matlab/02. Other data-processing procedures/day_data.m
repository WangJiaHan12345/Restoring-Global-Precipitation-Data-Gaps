%һ��Ϊ���ֱ�׼�Ľ�ˮ����
FolderPath=input('���������ݴ洢�ļ���:','s'); %����
index=strfind(FolderPath,'\');  %����ַ�'\'��FolderPath��λ��
SaveFolder=strcat('H:\ʱ��Ԥ��\24��\02_day_data\','cpc\ceshi'); %����ļ���·��
if exist(SaveFolder,'dir')~=7  %���·�����������½�·��
    mkdir(SaveFolder);
end
Files=dir(FolderPath);
FilesCount=length(Files);
disp('������...');

fid_2 = fopen('H:\ʱ��Ԥ��\24��\dem\DEM.txt');
data1 = cell2mat(textscan(fid_2,'%f','headerlines',6));
data1 = reshape(data1,720,229);
data1 = data1';
fclose(fid_2);  

parfor k=3:FilesCount
      FilePath=strcat(FolderPath,'\',Files(k).name);  %�ļ�·��\�ļ���
      Name=Files(k).name;
      location=strfind(Name,'.');  %����ַ�'.'��FilePath��λ��
  
      fid=fopen(FilePath,'rb','l');  % 'rb'�Զ����Ʒ�ʽֻ�����ʹ��ļ���Ҳ����ֱ��'r';'l':little endianС�����
      data = cell2mat(textscan(fid,'%f','headerlines',6));
      data = reshape(data,720,240);
      data = data';
      
      data2=zeros(235,1); 
      a=0;
       
       for i=1:1:229
           for j=1:1:720
               if data1(i,j)~=-9999
                   a=a+1;
                  data2(a,1)=data(i,j); 
               end
           end
       end
           
     SaveFiles=strcat(Name(location(end)-8:location(end)-1),'.txt'); %CPC����ļ���·��
     %SaveFiles=strcat(Name(1:location(end)-1),'.txt');
    
     outfile=strcat(SaveFolder,'\',SaveFiles);

     if exist(outfile,'file')~=0 
        delete(outfile);     
     end
     fid1=fopen(outfile,'w');
     
     
     for i=1:1:235
         for j=1:1:1
             if j==1
                 fprintf(fid1,'%g\r\n',data2(i,j));
             else
                fprintf(fid1,'%g ',data2(i,j));
             end
         end   
     end
     fclose(fid1); 
     fclose(fid); 

end
disp('�������')


 