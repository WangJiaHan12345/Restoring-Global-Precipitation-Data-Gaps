FolderPath=input('���������ݴ洢�ļ���:','s'); %����
index=strfind(FolderPath,'\');  %����ַ�'\'��FolderPath��λ��
SaveFolder=strcat('G:\���\global\����\','s'); %����ļ���·��
if exist(SaveFolder,'dir')~=7  %���·�����������½�·��
    mkdir(SaveFolder);
end
Files=dir(FolderPath);
FilesCount=length(Files);
disp('������...');

sum=zeros(1461,1);


for k=21463:FilesCount
          FilePath=strcat(FolderPath,'\',Files(k).name);  %�ļ�·��\�ļ���
          Name=Files(k).name;
          location=strfind(Name,'.');  %����ַ�'.'��FilePath��λ��

          fid=fopen(FilePath,'rb','l');  % 'rb'�Զ����Ʒ�ʽֻ�����ʹ��ļ���Ҳ����ֱ��'r';'l':little endianС�����
          data = cell2mat(textscan(fid,'%f','headerlines',0));
          data = reshape(data,1,1461);
          data = data';
          fclose(fid); 
          
          for i=1:1:1461
              sum(i,1)=sum(i,1)+ data(i,1);
          end
end
    
SaveFiles=strcat('cpc','.txt'); %CPC����ļ���·��
outfile=strcat(SaveFolder,'\',SaveFiles);

if exist(outfile,'file')~=0 
delete(outfile);     
end
fid1=fopen(outfile,'w');

days=8181;

for i=1:1:1461
 for j=1:1:1
     if j==1
         fprintf(fid1,'%g\r\n',sum(i,j)/days);
     else
         fprintf(fid1,'%g ',sum(i,j)/days);
     end
 end 
end
 fclose(fid1); 
 disp('�������');

% 
% FolderPath=input('���������ݴ洢�ļ���:','s'); %����
% index=strfind(FolderPath,'\');  %����ַ�'\'��FolderPath��λ��
% SaveFolder=strcat('G:\���\shirun\global\1-12\','����߶�'); %����ļ���·��
% if exist(SaveFolder,'dir')~=7  %���·�����������½�·��
%     mkdir(SaveFolder);
% end
% Files=dir(FolderPath);
% FilesCount=length(Files);
% disp('������...');
% 
% 
% sum=zeros(1461,7140);
% 
% for k=3:FilesCount
%           FilePath=strcat(FolderPath,'\',Files(k).name);  %�ļ�·��\�ļ���
%           Name=Files(k).name;
%           location=strfind(Name,'.');  %����ַ�'.'��FilePath��λ��
% 
%           fid=fopen(FilePath,'rb','l');  % 'rb'�Զ����Ʒ�ʽֻ�����ʹ��ļ���Ҳ����ֱ��'r';'l':little endianС�����
%           data = cell2mat(textscan(fid,'%f','headerlines',0));
%           data = reshape(data,1,1461);
%           data = data';
%           fclose(fid); 
%           
%           
%           for i=1:1:1461
%                 sum(i,k-2)=data(i,1);
%           end
% end
%     
% SaveFiles=strcat('NN','.txt'); %CPC����ļ���·��
% outfile=strcat(SaveFolder,'\',SaveFiles);
% 
% if exist(outfile,'file')~=0 
% delete(outfile);     
% end
% fid1=fopen(outfile,'w');
% 
% 
% for i=1:1:1461
%  for j=1:1:7140
%      if j==7140
%          fprintf(fid1,'%g\r\n',sum(i,j));
%      else
%          fprintf(fid1,'%g ',sum(i,j));
%      end
%  end 
% end
%  fclose(fid1); 
%  disp('�������');
% 
%   