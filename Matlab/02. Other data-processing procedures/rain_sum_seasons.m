FolderPath=input('���������ݴ洢�ļ���:','s'); %����
index=strfind(FolderPath,'\');  %����ַ�'\'��FolderPath��λ��
SaveFolder=strcat('G:\���\ganhan\global\','1-12'); %����ļ���·��
if exist(SaveFolder,'dir')~=7  %���·�����������½�·��
    mkdir(SaveFolder);
end
Files=dir(FolderPath);
FilesCount=length(Files);
disp('������...');

day=1461;
grid=4147;
rain=zeros(1,1);

for k=3:FilesCount
      FilePath=strcat(FolderPath,'\',Files(k).name);  %�ļ�·��\�ļ���
      Name=Files(k).name;
      location=strfind(Name,'.');  %����ַ�'.'��FilePath��λ��
  
      fid=fopen(FilePath,'rb','l');  % 'rb'�Զ����Ʒ�ʽֻ�����ʹ��ļ���Ҳ����ֱ��'r';'l':little endianС�����
      data = cell2mat(textscan(fid,'%f','headerlines',0));
      data = reshape(data,1,day);
      data = data';
      fclose(fid);
       
       for i=1:1:day
           for j=1:1:1
                  rain(1,1)=rain(1,1)+data(i,j);
           end
       end
end
       
SaveFiles=strcat('Final_rain_sum','.txt'); %CPC����ļ���·��
outfile=strcat(SaveFolder,'\',SaveFiles);

if exist(outfile,'file')~=0 
delete(outfile);     
end
fid1=fopen(outfile,'w');

     
 for i=1:1:1
     for j=1:1:1                
             fprintf(fid1,'%g\r\n',rain(i,j)/grid);
     end
 end  
fclose(fid1);


      
disp('�������')


 